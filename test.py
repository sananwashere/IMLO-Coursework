import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import PetResidualCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 160
BATCH_SIZE = 32
MODEL_PATH = "model.pth"
TTA_STEPS = 5


def main():
    print("Using device:", DEVICE)

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    tta_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMAGE_SIZE, padding=20),
    ])

    test_dataset = datasets.OxfordIIITPet(
        root="data",
        split="test",
        target_types="category",
        download=True,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = PetResidualCNN(num_classes=37).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(DEVICE)

            preds = torch.stack([
                model(tta_transform(images).to(DEVICE))
                for _ in range(TTA_STEPS)
            ]).mean(0)

            _, predicted = torch.max(preds, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total

    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()