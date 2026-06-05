"""MNIST / FashionMNIST 4-class subset DataLoader.

Each (x, y, idx) tuple includes the sample's *global* index within its split,
which is required by score.py (sample-wise score lookup) and by train.py
(WE-mask membership check).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

MNIST_CLASSES = (0, 3, 6, 8)
FMNIST_CLASSES = (0, 1, 2, 3)
CIFAR10_CLASSES = (0, 1, 5, 9)  # airplane, automobile, dog, truck


class IndexedSubset(Dataset):
    """Wraps a torchvision dataset, restricts to chosen classes & per-class
    sample counts, remaps labels to 0..C-1, and returns (x, y_remapped, idx).

    `idx` is the position within this subset (0..len-1), stable across epochs
    because we materialize the index list once at construction time.
    """

    def __init__(
        self,
        base: datasets.VisionDataset,
        classes: Sequence[int],
        n_per_class: int,
        seed: int,
    ):
        self.base = base
        self.classes = tuple(classes)
        self.label_map = {c: i for i, c in enumerate(self.classes)}

        targets = base.targets if isinstance(base.targets, torch.Tensor) else torch.tensor(base.targets)
        gen = torch.Generator().manual_seed(seed)
        chosen: list[int] = []
        for c in self.classes:
            idxs = (targets == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(idxs), generator=gen)
            picked = idxs[perm[:n_per_class]].tolist()
            chosen.extend(picked)
        self.indices = chosen
        self.remapped_labels = torch.tensor(
            [self.label_map[int(targets[i])] for i in chosen], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, _ = self.base[self.indices[idx]]
        y = int(self.remapped_labels[idx])
        return x, y, idx


def _load_raw(name: str) -> tuple[datasets.VisionDataset, datasets.VisionDataset]:
    tfm = transforms.Compose([transforms.ToTensor()])
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if name == "mnist":
        train = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=tfm)
        test = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=tfm)
    elif name == "fmnist":
        train = datasets.FashionMNIST(str(DATA_ROOT), train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST(str(DATA_ROOT), train=False, download=True, transform=tfm)
    elif name == "cifar10":
        tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 32x32x3 RGB -> 32x32x1
            transforms.ToTensor(),
        ])
        train = datasets.CIFAR10(str(DATA_ROOT), train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(str(DATA_ROOT), train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test


def load_data(
    name: str = "mnist",
    n_train_per_class: int = 200,
    n_val_per_class: int = 50,
    n_test_per_class: int = 100,
    batch_size: int = 32,
    seed: int = 42,
    classes: Sequence[int] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, IndexedSubset, IndexedSubset, IndexedSubset]:
    """Returns (train_loader, val_loader, test_loader, train_ds, val_ds, test_ds).

    Train and val are carved out of the torchvision *train* split (so they
    don't leak into test). Test is from the torchvision *test* split.

    `classes` overrides the default subset for the dataset (useful for the
    8-class MNIST experiment in Stage 2-6).
    """
    raw_train, raw_test = _load_raw(name)
    if classes is None:
        if name == "mnist":
            classes = MNIST_CLASSES
        elif name == "fmnist":
            classes = FMNIST_CLASSES
        elif name == "cifar10":
            classes = CIFAR10_CLASSES
        else:
            raise ValueError(f"No default classes for {name}")

    # Train: pull n_train+n_val per class from raw_train, then split.
    # Easiest deterministic split: pick n_train+n_val with one seed, then slice.
    pooled = IndexedSubset(raw_train, classes, n_train_per_class + n_val_per_class, seed=seed)

    # Per-class slice: first n_train → train, next n_val → val.
    train_local: list[int] = []
    val_local: list[int] = []
    counts: dict[int, int] = {c: 0 for c in range(len(classes))}
    for local_i, lbl in enumerate(pooled.remapped_labels.tolist()):
        if counts[lbl] < n_train_per_class:
            train_local.append(local_i)
        else:
            val_local.append(local_i)
        counts[lbl] += 1

    train_ds = _SliceWithReindex(pooled, train_local)
    val_ds = _SliceWithReindex(pooled, val_local)
    test_ds = IndexedSubset(raw_test, classes, n_test_per_class, seed=seed)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


class _SliceWithReindex(Dataset):
    """Slice of an IndexedSubset that reassigns idx to 0..len-1.

    We need this because IndexedSubset returns the *pooled* index, but score.py
    keys scores on each split's own 0-based index.
    """

    def __init__(self, parent: IndexedSubset, local_indices: list[int]):
        self.parent = parent
        self.local_indices = local_indices

    def __len__(self) -> int:
        return len(self.local_indices)

    def __getitem__(self, new_idx: int):
        x, y, _ = self.parent[self.local_indices[new_idx]]
        return x, y, new_idx


if __name__ == "__main__":
    # Sanity check
    tl, vl, te, tds, vds, teds = load_data("mnist", batch_size=8, seed=42)
    print(f"train={len(tds)}  val={len(vds)}  test={len(teds)}")
    xb, yb, ib = next(iter(tl))
    print(f"batch x.shape={tuple(xb.shape)}  y.shape={tuple(yb.shape)}  idx={ib.tolist()[:8]}")
    print(f"y unique={sorted(set(yb.tolist()))}  range x=[{xb.min():.3f},{xb.max():.3f}]")
    # class balance check
    all_y = torch.cat([torch.tensor([y for _, y, _ in [tds[i]]]) for i in range(len(tds))])
    counts = torch.bincount(all_y, minlength=4)
    print(f"train class counts: {counts.tolist()}")
