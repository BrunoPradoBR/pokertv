"""
Fine-tune ResNet18 on synthesized card images. Exports a TorchScript model.

Usage:
  python -m pokertv.training.train
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

DATA_DIR = Path("assets/training_data")
MODEL_OUT = Path("models/card_recognizer.pt")
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for name, param in net.named_parameters():
        if not any(name.startswith(p) for p in ("layer3", "layer4", "fc")):
            param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def train() -> None:
    dataset = ImageFolder(root=str(DATA_DIR), transform=TRAIN_TRANSFORM)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    val_set.dataset.transform = VAL_TRANSFORM

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    net = build_model(len(dataset.classes)).to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        net.train(True)
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        net.train(False)
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                val_correct += (net(images).argmax(1) == labels).sum().item()
                val_total += images.size(0)

        print(f"Epoch {epoch}/{EPOCHS} | loss={total_loss/total:.4f} | "
              f"train_acc={correct/total:.4f} | val_acc={val_correct/val_total:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(net.train(False))
    scripted.save(str(MODEL_OUT))
    print(f"Saved TorchScript model to {MODEL_OUT}")
    print(f"Class order: {dataset.classes}")


if __name__ == "__main__":
    train()
