"""Classical CNN teacher (LeNet-style) for 4-class MNIST/FashionMNIST.

Trained once per dataset and saved to results/teacher_<dataset>.pt. All KD
methods reuse the same teacher weights to keep comparisons fair.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_DIR = Path(__file__).resolve().parent.parent / "results"


class TeacherCNN(nn.Module):
    def __init__(self, num_classes: int = 4, input_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # After conv1+pool: input_size//2 ; after conv2 (5x5 no pad): -4 ; after pool: //2
        s = (input_size // 2 - 4) // 2
        self.fc1 = nn.Linear(16 * s * s, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_teacher(
    train_loader,
    val_loader,
    num_classes: int = 4,
    n_epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cpu",
    input_size: int = 28,
) -> tuple[TeacherCNN, dict]:
    model = TeacherCNN(num_classes, input_size=input_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_acc": []}

    for ep in range(n_epochs):
        model.train()
        total = 0.0
        n = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        train_loss = total / max(n, 1)

        model.eval()
        correct = 0
        seen = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                correct += (pred == y).sum().item()
                seen += y.size(0)
        val_acc = correct / max(seen, 1)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        print(f"  [teacher] ep {ep + 1:2d}/{n_epochs}  loss={train_loss:.4f}  val_acc={val_acc:.3f}")
    return model, history


def get_or_train_teacher(
    dataset: str,
    train_loader,
    val_loader,
    num_classes: int = 4,
    n_epochs: int = 15,
    device: str = "cpu",
    force_retrain: bool = False,
) -> TeacherCNN:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"teacher_{dataset}.pt"
    model = TeacherCNN(num_classes).to(device)
    if ckpt.exists() and not force_retrain:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        print(f"  [teacher] loaded from {ckpt}")
        return model
    print(f"  [teacher] training fresh ({n_epochs} epochs)...")
    model, _ = train_teacher(train_loader, val_loader, num_classes, n_epochs, device=device)
    torch.save(model.state_dict(), ckpt)
    print(f"  [teacher] saved to {ckpt}")
    return model


if __name__ == "__main__":
    # Sanity check: forward pass shape, then a 2-epoch quick train
    from data import load_data

    tl, vl, _, _, _, _ = load_data("mnist", batch_size=32, seed=42)
    m = TeacherCNN(num_classes=4)
    xb, _, _ = next(iter(tl))
    out = m(xb)
    print(f"forward shape: x={tuple(xb.shape)} -> logit={tuple(out.shape)}")
    assert out.shape == (xb.size(0), 4), "logit shape mismatch"

    # 2-epoch dry-run to confirm training loop works
    m, hist = train_teacher(tl, vl, n_epochs=2)
    print(f"2-epoch dry run: val_acc={hist['val_acc'][-1]:.3f}")
