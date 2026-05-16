import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from model import PetCNN


# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# General settings
DATA_ROOT = "data"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
MODEL_PATH = "model.pth"

# Curriculum learning settings
CURRICULUM_END = 20
MIN_CROP_PROB = 0.0

# Fixed seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)


class OxfordPetTrimapDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        augment=False,
        crop_prob=0.0,
        margin=0.15
    ):
        self.root = root
        self.split = split
        self.augment = augment
        self.crop_prob = crop_prob
        self.margin = margin

        self.base_dir = os.path.join(root, "oxford-iiit-pet")
        self.images_dir = os.path.join(self.base_dir, "images")
        self.xml_dir = os.path.join(self.base_dir, "annotations", "xmls")
        self.trimap_dir = os.path.join(self.base_dir, "annotations", "trimaps")
        self.split_file = os.path.join(self.base_dir, "annotations", f"{split}.txt")

        self.color_jitter = transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        )

        self.random_erasing = transforms.RandomErasing(p=0.15)

        self.samples = []

        with open(self.split_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) >= 2:
                    image_name = parts[0]
                    label = int(parts[1]) - 1
                    self.samples.append((image_name, label))

    def __len__(self):
        return len(self.samples)

    def crop_to_roi(self, image, trimap, image_name):
        xml_path = os.path.join(self.xml_dir, f"{image_name}.xml")

        if not os.path.exists(xml_path):
            return image, trimap

        try:
            root = ET.parse(xml_path).getroot()
            box = root.find(".//bndbox")

            if box is None:
                return image, trimap

            width, height = image.size

            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)

            box_width = xmax - xmin
            box_height = ymax - ymin

            pad_x = int(box_width * self.margin)
            pad_y = int(box_height * self.margin)

            xmin = max(0, xmin - pad_x)
            ymin = max(0, ymin - pad_y)
            xmax = min(width, xmax + pad_x)
            ymax = min(height, ymax + pad_y)

            if xmax <= xmin or ymax <= ymin:
                return image, trimap

            image = image.crop((xmin, ymin, xmax, ymax))
            trimap = trimap.crop((xmin, ymin, xmax, ymax))

            return image, trimap

        except Exception:
            return image, trimap

    def trimap_to_mask(self, trimap):
        trimap_np = np.array(trimap, dtype=np.float32)

        mask = np.zeros_like(trimap_np, dtype=np.float32)

        # Oxford trimap values:
        # 1 = pet foreground, 2 = boundary, 3 = background
        mask[trimap_np == 1] = 1.0
        mask[trimap_np == 2] = 0.5
        mask[trimap_np == 3] = 0.0

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return mask_tensor

    def __getitem__(self, index):
        image_name, label = self.samples[index]

        image_path = os.path.join(self.images_dir, f"{image_name}.jpg")
        trimap_path = os.path.join(self.trimap_dir, f"{image_name}.png")

        image = Image.open(image_path).convert("RGB")
        trimap = Image.open(trimap_path)

        # ROI crop during training only, controlled by curriculum
        if random.random() < self.crop_prob:
            image, trimap = self.crop_to_roi(image, trimap, image_name)

        # Resize both image and trimap
        image = TF.resize(image, (IMAGE_SIZE + 20, IMAGE_SIZE + 20))
        trimap = TF.resize(
            trimap,
            (IMAGE_SIZE + 20, IMAGE_SIZE + 20),
            interpolation=TF.InterpolationMode.NEAREST
        )

        # Joint random crop
        if self.augment:
            i, j, h, w = transforms.RandomCrop.get_params(
                image,
                output_size=(IMAGE_SIZE, IMAGE_SIZE)
            )

            image = TF.crop(image, i, j, h, w)
            trimap = TF.crop(trimap, i, j, h, w)
        else:
            image = TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            trimap = TF.resize(
                trimap,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=TF.InterpolationMode.NEAREST
            )

        # Joint horizontal flip
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            trimap = TF.hflip(trimap)

        # Joint rotation
        if self.augment:
            angle = random.uniform(-10, 10)

            image = TF.rotate(image, angle)
            trimap = TF.rotate(
                trimap,
                angle,
                interpolation=TF.InterpolationMode.NEAREST
            )

        # Image-only colour augmentations
        if self.augment:
            image = self.color_jitter(image)

            if random.random() < 0.05:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

        # RGB image tensor
        image_tensor = TF.to_tensor(image)

        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Trimap mask tensor
        mask_tensor = self.trimap_to_mask(trimap)

        # Combine RGB + trimap mask
        combined = torch.cat([image_tensor, mask_tensor], dim=0)

        if self.augment:
            combined = self.random_erasing(combined)

        return combined, label


def main():
    print("Using device:", DEVICE)

    # Download/check dataset exists
    datasets.OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",
        target_types="category",
        download=True
    )

    train_dataset = OxfordPetTrimapDataset(
        root=DATA_ROOT,
        split="trainval",
        augment=True,
        crop_prob=1.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"Training images: {len(train_dataset)}")

    model = PetCNN(num_classes=37).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimiser = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=1e-3,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader)
    )

    for epoch in range(EPOCHS):
        # Curriculum: reduce ROI crop over time
        if epoch < CURRICULUM_END:
            progress = epoch / (CURRICULUM_END - 1)
            crop_prob = 1.0 - progress * (1.0 - MIN_CROP_PROB)
        else:
            crop_prob = MIN_CROP_PROB

        train_dataset.crop_prob = crop_prob

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimiser.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimiser.step()
            scheduler.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Crop Prob: {crop_prob:.2f} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("Training complete")
    print("Saved final model")


if __name__ == "__main__":
    main()