import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import PetResidualCNN


# Use GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main training settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
MODEL_PATH = "model.pth"

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def calculate_accuracy(model, loader):
    # Put model in evaluation mode
    model.eval()

    correct = 0
    total = 0

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Get model predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    print("Using device:", DEVICE)

    # Training transforms with data augmentation
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

    # Validation transforms without random augmentation
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Dataset used for training
    full_train_dataset = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transform
    )

    # Same dataset used for validation, but with validation transforms
    full_val_dataset = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=True,
        transform=val_transform
    )

    # Create a fixed random split
    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(full_train_dataset),
        generator=generator
    ).tolist()

    # Use 80% for training and 20% for validation
    train_size = int(0.8 * len(indices))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    # Load training data in batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Load validation data in batches
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create the model
    model = PetResidualCNN(num_classes=37).to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimiser
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader)
    )

    best_val_acc = 0.0

    # Training loop
    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation and parameter update
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Calculate validation accuracy
        val_acc = calculate_accuracy(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("Saved best model")

    print("Training complete")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()