"""
Evaluate card recognizer accuracy and print per-class report.

Usage:
  python -m pokertv.training.evaluate
"""
from __future__ import annotations
from pathlib import Path
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

DATA_DIR = Path("assets/training_data")
MODEL_PATH = Path("models/card_recognizer.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def evaluate() -> None:
    dataset = ImageFolder(root=str(DATA_DIR), transform=TRANSFORM)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    classes = dataset.classes
    n = len(classes)

    model = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)
    model.train(False)

    confusion = torch.zeros(n, n, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            for t, p in zip(labels, preds):
                confusion[t, p] += 1

    correct = confusion.diag().sum().item()
    total = confusion.sum().item()
    print(f"\nOverall accuracy: {correct/total:.4f} ({correct}/{total})")
    print("\nPer-class accuracy:")
    for i, cls in enumerate(classes):
        row_total = confusion[i].sum().item()
        acc = confusion[i, i].item() / row_total if row_total > 0 else 0.0
        print(f"  {cls:4s}: {acc:.4f}  ({confusion[i,i].item()}/{row_total})")


if __name__ == "__main__":
    evaluate()
