"""Compute π_neg for the 8q+10c setup on MNIST and FMNIST (paper §RQ4 footnote).

Mirrors the π_neg block in build_main_table.py: build a tutorial-reducer
QuantumStudent (untrained reference encoder, n_layers=4), score the training
split against training-derived class means, report Pr[HEM(x) < 0].
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(".").resolve() / "src"))
from data import load_data
from student import QuantumStudent
from score import compute_class_means, compute_scores

DATA_SEED = 42

SETUPS = [
    ("MNIST",  "mnist",  8, 10, tuple(range(10))),
    ("FMNIST", "fmnist", 8, 10, tuple(range(10))),
]

for ds, dskey, nq, nc, cls in SETUPS:
    tl, vl, te, _, _, _ = load_data(dskey, 400, 100, 200, 32, DATA_SEED, cls)
    torch.manual_seed(DATA_SEED)
    ref = QuantumStudent(n_qubits=nq, n_layers=4, num_classes=nc,
                         reducer_type="tutorial", device_kind="default")
    cm = compute_class_means(ref.reducer, tl, nq, nc)
    tr_sc = compute_scores(ref.reducer, tl, cm, nq, nc)
    tr_vals = np.array(list(tr_sc.values()))
    pi_neg = (tr_vals < 0).mean()
    print(f"{ds} 8q10c  pi_neg = {pi_neg*100:.1f}%  "
          f"(n_train={len(tr_vals)}, score min={tr_vals.min():.3f}, "
          f"max={tr_vals.max():.3f}, mean={tr_vals.mean():.3f})")
