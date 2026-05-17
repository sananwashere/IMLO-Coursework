import os
import numpy as np
from PIL import Image

import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms.functional as TF

from model import PetCNN


# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "data"
IMAGE_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "model.pth"


class OxfordPetTrimapDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split

        self.base_dir = os.path.join(root, "oxford-iiit-pet")
        self.images_dir = os.path.join(self.base_dir, "images")
        self.trimap_dir = os.path.join(self.base_dir, "annotations", "trimaps")
        self.split_file = os.path.join(self.base_dir, "annotations", f"{split}.txt")

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


        image = TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        trimap = TF.resize(
            trimap,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=TF.InterpolationMode.NEAREST
        )

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

        return combined, label


def main():
    print("Using device:", DEVICE)

    # Download and check official test split exists
    datasets.OxfordIIITPet(
        root=DATA_ROOT,
        split="test",
        target_types="category",
        download=True
    )

    test_dataset = OxfordPetTrimapDataset(
        root=DATA_ROOT,
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = PetCNN(num_classes=37).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total

    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()