import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import PetResidualCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 160
BATCH_SIZE = 32
MODEL_PATH = "model.pth"


def main():
    print("Using device:", DEVICE)

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
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
        num_workers=0,
        pin_memory=True
    )

    class_names = test_dataset.classes
    num_classes = len(class_names)

    model = PetResidualCNN(num_classes=37).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            for true_label, predicted_label in zip(labels.cpu(), predictions.cpu()):
                true_label = true_label.item()
                predicted_label = predicted_label.item()

                total_per_class[true_label] += 1
                confusion_matrix[true_label][predicted_label] += 1

                if true_label == predicted_label:
                    correct_per_class[true_label] += 1
                    total_correct += 1

                total_images += 1

    overall_accuracy = 100 * total_correct / total_images

    print()
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
    print()

    print("Per-class accuracy:")
    print("-" * 60)

    class_accuracies = []

    for i in range(num_classes):
        if total_per_class[i] > 0:
            accuracy = 100 * correct_per_class[i] / total_per_class[i]
        else:
            accuracy = 0

        class_accuracies.append((class_names[i], accuracy, int(total_per_class[i])))

        print(f"{class_names[i]:25s} | {accuracy:6.2f}% | {int(total_per_class[i])} images")

    print()
    print("Worst 10 classes:")
    print("-" * 60)

    class_accuracies.sort(key=lambda x: x[1])

    for class_name, accuracy, total in class_accuracies[:10]:
        print(f"{class_name:25s} | {accuracy:6.2f}% | {total} images")

    print()
    print("Best 10 classes:")
    print("-" * 60)

    for class_name, accuracy, total in class_accuracies[-10:][::-1]:
        print(f"{class_name:25s} | {accuracy:6.2f}% | {total} images")

    print()
    print("Most common confusions:")
    print("-" * 60)

    confusions = []

    for true_class in range(num_classes):
        for predicted_class in range(num_classes):
            if true_class != predicted_class:
                count = confusion_matrix[true_class][predicted_class]

                if count > 0:
                    confusions.append(
                        (
                            count,
                            class_names[true_class],
                            class_names[predicted_class]
                        )
                    )

    confusions.sort(reverse=True)

    for count, true_name, predicted_name in confusions[:20]:
        print(f"{true_name:25s} predicted as {predicted_name:25s} | {count} times")


if __name__ == "__main__":
    main()