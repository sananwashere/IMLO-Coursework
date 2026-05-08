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

from model import PetResidualCNN


# Use GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# General Settings
DATA_ROOT = "data"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
MODEL_PATH = "model.pth"

# Curriculum learning settings
CURRICULUM_END = 20
MIN_CROP_PROB = 0.0

# TRIMAP Probabilities
MAX_TRIMAP_PROB = 0.6
MIN_TRIMAP_PROB = 0.0

# Fixed seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)


# dataset that can use ROI boxes and trimaps during training
class OxfordPetROIDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        transform=None,
        crop_prob=0.0,
        trimap_prob=0.0,
        margin=0.15
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.crop_prob = crop_prob
        self.trimap_prob = trimap_prob
        self.margin = margin

        self.base_dir = os.path.join(root, "oxford-iiit-pet")
        self.images_dir = os.path.join(self.base_dir, "images")
        self.xml_dir = os.path.join(self.base_dir, "annotations", "xmls")
        self.trimap_dir = os.path.join(self.base_dir, "annotations", "trimaps")
        self.split_file = os.path.join(self.base_dir, "annotations", f"{split}.txt")

        self.samples = []

        # Read image names and labels from split file
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
        # Use XML bounding box to crop the pet/head region
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

            # Add small margin around the box
            pad_x = int(box_width * self.margin)
            pad_y = int(box_height * self.margin)

            xmin = max(0, xmin - pad_x)
            ymin = max(0, ymin - pad_y)
            xmax = min(width, xmax + pad_x)
            ymax = min(height, ymax + pad_y)

            if xmax <= xmin or ymax <= ymin:
                return image, trimap

            image = image.crop((xmin, ymin, xmax, ymax))

            # Crop trimap the same way if it exists
            if trimap is not None:
                trimap = trimap.crop((xmin, ymin, xmax, ymax))

            return image, trimap

        except Exception:
            return image, trimap

    def apply_trimap_mask(self, image, trimap):
        # Use trimap to fade the background during training
        if trimap is None:
            return image

        image_np = np.array(image).astype(np.float32)
        trimap_np = np.array(trimap)

        mask = np.ones_like(trimap_np, dtype=np.float32) * 0.35

        # Oxford trimap values below
        # 1 = pet foreground, 2 = boundary, 3 = background
        mask[trimap_np == 1] = 1.0
        mask[trimap_np == 2] = 0.7
        mask[trimap_np == 3] = 0.35

        mask = np.expand_dims(mask, axis=2)

        background_colour = np.ones_like(image_np) * 127.0

        masked_image = image_np * mask + background_colour * (1.0 - mask)
        masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)

        return Image.fromarray(masked_image)

    def __getitem__(self, index):
        image_name, label = self.samples[index]

        image_path = os.path.join(self.images_dir, f"{image_name}.jpg")
        trimap_path = os.path.join(self.trimap_dir, f"{image_name}.png")

        image = Image.open(image_path).convert("RGB")

        trimap = None
        if os.path.exists(trimap_path):
            trimap = Image.open(trimap_path)

        # Apply ROI crop depending on current crop probability
        if random.random() < self.crop_prob:
            image, trimap = self.crop_to_roi(image, trimap, image_name)

        # Apply trimap mask depending on current trimap probability
        if random.random() < self.trimap_prob:
            image = self.apply_trimap_mask(image, trimap)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def main():
    print("Using device:", DEVICE)

    # Download and check the dataset exists
    datasets.OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",
        target_types="category",
        download=True
    )

    # Training augmentations
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.15),
    ])

    # full trainval split for training
    train_dataset = OxfordPetROIDataset(
        root=DATA_ROOT,
        split="trainval",
        transform=train_transform,
        crop_prob=1.0,
        trimap_prob=MAX_TRIMAP_PROB
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"Training images: {len(train_dataset)}")

    # Create model
    model = PetResidualCNN(num_classes=37).to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimiser
    optimiser = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=1e-3,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader)
    )

    for epoch in range(EPOCHS):
        # Curriculum used to reduce ROI crop and trimap masking over time
        if epoch < CURRICULUM_END:
            progress = epoch / (CURRICULUM_END - 1)

            crop_prob = 1.0 - progress * (1.0 - MIN_CROP_PROB)
            trimap_prob = MAX_TRIMAP_PROB - progress * (MAX_TRIMAP_PROB - MIN_TRIMAP_PROB)
        else:
            crop_prob = MIN_CROP_PROB
            trimap_prob = MIN_TRIMAP_PROB

        train_dataset.crop_prob = crop_prob
        train_dataset.trimap_prob = trimap_prob

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
            f"Trimap Prob: {trimap_prob:.2f} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

    # Save final trained model
    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("Training complete")
    print("Saved final model")


if __name__ == "__main__":
    main()