"""
train.py
Trains the SignGestureClassifier on landmark sequences from dataset.py.

Usage:
    python src/train.py --epochs 30 --batch-size 16
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import ISLGestureDataset
from model import SignGestureClassifier


def train(epochs: int = 30, batch_size: int = 16, lr: float = 1e-3,
          checkpoint_path: str = "models/gesture_classifier.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = ISLGestureDataset()
    if len(dataset) == 0:
        raise RuntimeError(
            "No training data found in data/processed/. See dataset.py for "
            "the expected layout, or point it at an open ISL dataset."
        )

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SignGestureClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sequences.size(0)

        train_loss = running_loss / len(train_ds)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved new best checkpoint (val_acc={val_acc:.3f}) -> {checkpoint_path}")

    print(f"Training complete. Best val_acc={best_val_acc:.3f}")


def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
