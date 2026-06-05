"""Train Baseline (no KD, 4L) under angle encoding across 6 setups for the RQ1 figure.

The main run scripts use amplitude encoding (Stage A: avg-pool + L2 normalize ->
state amplitudes). To show that the HEM-based CE/PE partition holds under a
different embedding too, this script trains the same Baseline architecture under
qml.AngleEmbedding(rotation='Y'). HEM scores and the 20th-percentile CE threshold
are computed under the same angle encoding.

Five random seeds are drawn at startup (same convention as run_paper_main{,_fmnist}.py)
so figure_rq1.py panel (c) can plot mean +/- std bars consistent with the amplitude side.

Output: results/q0_angle_baseline/results.json - one record per (setup, seed)
with the per-seed test metrics. Consumed by figure_rq1.py panel (c).
"""
from __future__ import annotations
import sys, json, time, random
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data import load_data
from teacher import TeacherCNN, train_teacher
from student import QuantumStudent
from score import compute_class_means, compute_scores, make_we_mask, apply_threshold
from train import train_method, evaluate

DATA_SEED = 42                                # fixed: controls which samples are drawn
ALL_SEEDS = random.sample(range(10000), 5)    # 5 random model-init / shuffle seeds
N_EPOCHS = 50
LR = 1e-3

SETUPS = [
    # (dataset, tag, n_qubits, num_classes, classes)
    ("mnist",  "4q4c",  4,  4, (0, 3, 6, 8)),
    ("mnist",  "4q10c", 4, 10, tuple(range(10))),
    ("mnist",  "8q4c",  8,  4, (0, 3, 6, 8)),
    ("fmnist", "4q4c",  4,  4, (0, 1, 5, 9)),
    ("fmnist", "4q10c", 4, 10, tuple(range(10))),
    ("fmnist", "8q4c",  8,  4, (0, 1, 5, 9)),
]

OUT = Path("results/q0_angle_baseline")
OUT.mkdir(parents=True, exist_ok=True)
print(f"# angle Baseline - {len(ALL_SEEDS)} seeds x {len(SETUPS)} setups")
print(f"# seeds (random this run): {ALL_SEEDS}")

results = []
t_start = time.perf_counter()
for ds, tag, nq, nc, cls in SETUPS:
    print(f"\n== {ds} {tag} (angle encoding) ==")
    tl, vl, te, _, _, _ = load_data(ds, 400, 100, 200, 32, DATA_SEED, cls)

    # Teacher (same as amplitude side - operates on raw 28x28, independent of student encoding)
    teacher = TeacherCNN(num_classes=nc)
    print(f"  training teacher...")
    teacher, _ = train_teacher(tl, vl, num_classes=nc, n_epochs=15)
    teacher.eval()

    # HEM scores under angle encoding -> CE mask (deterministic given DATA_SEED)
    torch.manual_seed(DATA_SEED)
    ref = QuantumStudent(n_qubits=nq, n_layers=4, num_classes=nc,
                         reducer_type="angle", device_kind="default", encoding="angle")
    cm = compute_class_means(ref.reducer, tl, nq, nc, encoding="angle")
    tr_sc = compute_scores(ref.reducer, tl, cm, nq, nc, encoding="angle")
    te_sc = compute_scores(ref.reducer, te, cm, nq, nc, encoding="angle")
    train_we, thr = make_we_mask(tr_sc, percentile=20.0)
    test_we = apply_threshold(te_sc, thr)
    print(f"  CE: train={len(train_we)} test={len(test_we)} thr={thr:.4f}")

    for seed in ALL_SEEDS:
        torch.manual_seed(seed)
        student = QuantumStudent(n_qubits=nq, n_layers=4, num_classes=nc,
                                 reducer_type="angle", device_kind="default", encoding="angle")
        t0 = time.perf_counter()
        train_method(method_id=1, student=student, teacher=teacher,
                     train_loader=tl, val_loader=vl,
                     train_we_mask=train_we, val_we_mask=test_we,
                     n_epochs=N_EPOCHS, lr=LR, log_every=N_EPOCHS,
                     lambda_kd=0.0, temperature=1.0)
        elapsed = time.perf_counter() - t0
        student.eval()
        m = evaluate(student, te, test_we)
        print(f"    seed={seed}  {elapsed:.0f}s  all={m['acc_all']:.3f} CE={m['acc_we']:.3f} PE={m['acc_non_we']:.3f}")

        results.append({
            "dataset": ds, "tag": tag, "n_qubits": nq, "num_classes": nc,
            "encoding": "angle", "seed": seed, "ce_threshold": thr,
            "test": m,
        })

(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
print(f"\n# saved {OUT/'results.json'}  ({time.perf_counter()-t_start:.0f}s total)")
