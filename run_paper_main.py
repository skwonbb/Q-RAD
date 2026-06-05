"""Q-RAD main experiments on MNIST — 5 seeds × 3 setups × 5 models.

Method mapping (paper label ↔ internal method_id):
  Baseline       = method_id 1, no KD                          (4L)
  Baseline_KD    = method_id 3, uniform KD,     λ=0.1, T=2     (4L)
  Fair           = method_id 1, no KD                          (8L, parameter-matched)
  PQC_PE         = method_id 6, KD on PE only,  λ=0.1, T=2     (4L specialist)
  PQC_CE_lam0p3  = method_id 8, KD on CE only,  λ=0.3, T=2     (4L specialist)
  Q-RAD          = oracle-routed combination of PQC_PE and PQC_CE

CE threshold = bottom 20% of HEM scores on training set.
Outputs to results/paper/<tag>/seed_<seed>/<Baseline|...>.{json,pt}.
Skips seeds whose outputs already exist.
"""
from __future__ import annotations
import sys, time, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data import load_data
from teacher import TeacherCNN, train_teacher
from student import QuantumStudent
from score import compute_class_means, compute_scores, make_we_mask, apply_threshold
from train import train_method, evaluate

DATA_SEED = 42                          # fixed: controls which samples are drawn for train/val/test
ALL_SEEDS = random.sample(range(10000), 5)  # 5 random model-init / shuffle seeds drawn per run
N_TRAIN_PER_CLASS = 400
N_VAL_PER_CLASS = 100
N_TEST_PER_CLASS = 200
BATCH_SIZE = 32
LR = 1e-3
N_EPOCHS = 50
N_LAYERS = 4
N_LAYERS_FAIR = 8
PERCENTILE = 20.0
BASELINE_KD_LAM = 0.1; BASELINE_KD_T = 2.0
PQC_PE_LAM = 0.1; PQC_PE_T = 2.0
PQC_CE_LAM = 0.3; PQC_CE_T = 2.0

# tag, n_qubits, num_classes, classes
SETUPS = [
    ("4q4c",  4,  4, (0, 3, 6, 8)),
    ("4q10c", 4, 10, tuple(range(10))),
    ("8q4c",  8,  4, (0, 3, 6, 8)),
]
PAPER_DIR = Path("results/paper")

def per_sample_correct(model, loader):
    model.eval(); out = {}
    with torch.no_grad():
        for x, y, idx in loader:
            logits = model(x); probs = torch.softmax(logits, dim=-1)
            max_p, pred = probs.max(dim=-1)
            for i in range(x.size(0)):
                ii = int(idx[i].item())
                out[ii] = {"correct": int(pred[i] == y[i]), "pred": int(pred[i].item()),
                            "max_p": float(max_p[i].item()), "true": int(y[i].item())}
    return out

def route_oracle(p_pe, p_ce, test_ce_mask):
    """Oracle routing: send CE samples to PQC_CE, PE samples to PQC_PE."""
    p_pe = {str(k): v for k, v in p_pe.items()}; p_ce = {str(k): v for k, v in p_ce.items()}
    n_all = n_ce = n_pe = c_all = c_ce = c_pe = 0
    for k in p_pe:
        v_pe, v_ce = p_pe[k], p_ce[k]; in_ce = int(k) in test_ce_mask
        ok = v_ce["correct"] if in_ce else v_pe["correct"]
        n_all += 1; c_all += ok
        if in_ce: n_ce += 1; c_ce += ok
        else:     n_pe += 1; c_pe += ok
    return c_all/n_all, c_ce/n_ce if n_ce else 0, c_pe/n_pe if n_pe else 0

def train_and_save(seed_dir, seed, label, n_qubits, num_classes, n_layers, method_id,
                    kd_kwargs, teacher, train_loader, val_loader, train_we, val_we,
                    test_loader, test_we):
    p = seed_dir / f"{label}.json"; pt = seed_dir / f"{label}.pt"
    if p.exists() and pt.exists():
        return json.loads(p.read_text())
    torch.manual_seed(seed)
    student = QuantumStudent(n_qubits=n_qubits, n_layers=n_layers, num_classes=num_classes,
                              reducer_type="tutorial", device_kind="default")
    t0 = time.perf_counter()
    train_method(method_id=method_id, student=student, teacher=teacher,
                  train_loader=train_loader, val_loader=val_loader,
                  train_we_mask=train_we, val_we_mask=val_we,
                  n_epochs=N_EPOCHS, lr=LR, log_every=N_EPOCHS, **kd_kwargs)
    elapsed = time.perf_counter() - t0
    student.eval()
    test_metrics = evaluate(student, test_loader, test_we)
    per_idx = per_sample_correct(student, test_loader)
    res = {"test": test_metrics, "per_idx": per_idx, "elapsed": elapsed,
            "method_id": method_id, **kd_kwargs}
    seed_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    torch.save(student.state_dict(), pt)
    print(f"      [{label}] {elapsed:.0f}s  all={test_metrics['acc_all']:.3f} "
          f"WE={test_metrics['acc_we']:.3f} nW={test_metrics['acc_non_we']:.3f}")
    return res

def run_setup(tag, n_qubits, num_classes, classes):
    print(f"\n{'#'*100}\n# {tag}\n{'#'*100}")
    setup_dir = PAPER_DIR / tag
    setup_dir.mkdir(parents=True, exist_ok=True)

    tl, vl, te, train_ds, _, _ = load_data(
        "mnist", n_train_per_class=N_TRAIN_PER_CLASS, n_val_per_class=N_VAL_PER_CLASS,
        n_test_per_class=N_TEST_PER_CLASS, batch_size=BATCH_SIZE,
        seed=DATA_SEED, classes=classes,
    )
    # Teacher: load if cached, otherwise train fresh
    teacher_dst = setup_dir / "teacher.pt"
    teacher = TeacherCNN(num_classes=num_classes)
    if teacher_dst.exists():
        teacher.load_state_dict(torch.load(teacher_dst, map_location="cpu"))
    else:
        print(f"  [teacher] training fresh ({num_classes}-class, 15 epochs)...")
        teacher, _ = train_teacher(tl, vl, num_classes=num_classes, n_epochs=15)
        torch.save(teacher.state_dict(), teacher_dst)
    teacher.eval()
    print(f"  [teacher] acc={evaluate(teacher, te, set())['acc_all']:.3f}")

    # Scores + masks
    torch.manual_seed(DATA_SEED)
    ref = QuantumStudent(n_qubits=n_qubits, n_layers=N_LAYERS, num_classes=num_classes,
                          reducer_type="tutorial", device_kind="default")
    class_means = compute_class_means(ref.reducer, tl, n_qubits, num_classes)
    train_scores = compute_scores(ref.reducer, tl, class_means, n_qubits, num_classes)
    val_scores = compute_scores(ref.reducer, vl, class_means, n_qubits, num_classes)
    test_scores = compute_scores(ref.reducer, te, class_means, n_qubits, num_classes)
    train_we, thr = make_we_mask(train_scores, percentile=PERCENTILE)
    val_we = apply_threshold(val_scores, thr)
    test_we = apply_threshold(test_scores, thr)
    print(f"  CE: train={len(train_we)}/{len(train_ds)}  test={len(test_we)}/{len(te.dataset)}  thr={thr:.4f}")

    for seed in ALL_SEEDS:
        print(f"\n  --- seed {seed} ---")
        seed_dir = setup_dir / f"seed_{seed}"

        # Baseline: full data, no KD (4 layers)
        train_and_save(seed_dir, seed, "Baseline", n_qubits, num_classes, N_LAYERS, 1,
                        dict(lambda_kd=0.0, temperature=1.0),
                        teacher, tl, vl, train_we, val_we, te, test_we)
        # Baseline_KD: uniform KD (4 layers)
        train_and_save(seed_dir, seed, "Baseline_KD", n_qubits, num_classes, N_LAYERS, 3,
                        dict(lambda_kd=BASELINE_KD_LAM, temperature=BASELINE_KD_T),
                        teacher, tl, vl, train_we, val_we, te, test_we)
        # Fair: parameter-matched single PQC (8 layers, no KD)
        train_and_save(seed_dir, seed, "Fair", n_qubits, num_classes, N_LAYERS_FAIR, 1,
                        dict(lambda_kd=0.0, temperature=1.0),
                        teacher, tl, vl, train_we, val_we, te, test_we)
        # PQC_PE: task on all + KD on PE only, λ=0.1, T=2 (4 layers)
        g = torch.Generator().manual_seed(seed)
        full_loader = DataLoader(Subset(train_ds, list(range(len(train_ds)))),
                                  batch_size=BATCH_SIZE, shuffle=True, generator=g)
        train_and_save(seed_dir, seed, "PQC_PE", n_qubits, num_classes, N_LAYERS, 6,
                        dict(lambda_kd=PQC_PE_LAM, temperature=PQC_PE_T),
                        teacher, full_loader, vl, train_we, val_we, te, test_we)
        # PQC_CE_lam0p3: task on all + KD on CE only, λ=0.3, T=2 (4 layers)
        g = torch.Generator().manual_seed(seed)
        full_loader = DataLoader(Subset(train_ds, list(range(len(train_ds)))),
                                  batch_size=BATCH_SIZE, shuffle=True, generator=g)
        train_and_save(seed_dir, seed, "PQC_CE_lam0p3", n_qubits, num_classes, N_LAYERS, 8,
                        dict(lambda_kd=PQC_CE_LAM, temperature=PQC_CE_T),
                        teacher, full_loader, vl, train_we, val_we, te, test_we)

    # Aggregate setup summary
    setup_res = []
    models = ["Baseline", "Baseline_KD", "Fair", "PQC_PE", "PQC_CE_lam0p3"]
    for seed in ALL_SEEDS:
        sd = setup_dir / f"seed_{seed}"
        d = {m: json.load(open(sd/f"{m}.json")) for m in models}
        q_rad = route_oracle(d["PQC_PE"]["per_idx"], d["PQC_CE_lam0p3"]["per_idx"], test_we)
        setup_res.append({
            "seed": seed,
            **{m: d[m]["test"] for m in d},
            "Q-RAD": {"acc_all": q_rad[0], "acc_ce": q_rad[1], "acc_pe": q_rad[2]},
        })
    summary = {"tag": tag, "n_qubits": n_qubits, "num_classes": num_classes,
                "ce_threshold": thr, "config": "PQC_PE(λ=0.1,T=2) + PQC_CE_lam0p3(λ=0.3,T=2)",
                "results_per_seed": setup_res}
    (setup_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return setup_res

def main():
    print(f"# Q-RAD MNIST main run — {len(ALL_SEEDS)} seeds × {len(SETUPS)} setups × 5 models")
    print(f"# config: PQC_PE (λ=0.1, T=2) + PQC_CE_lam0p3 (λ=0.3, T=2), CE percentile=20%")
    print(f"# seeds (random this run): {ALL_SEEDS}")
    print(f"# saves to {PAPER_DIR}/<tag>/seed_<seed>/<Baseline|...>.{{json,pt}}")

    all_results = {}
    t0 = time.perf_counter()
    for tag, n_qubits, num_classes, classes in SETUPS:
        all_results[tag] = run_setup(tag, n_qubits, num_classes, classes)
    elapsed = time.perf_counter() - t0
    print(f"\n\n# all done. Total {elapsed/60:.1f}min")

    # Final cross-setup table
    print(f"\n{'='*110}")
    print(f"# Q-RAD MNIST final table — {len(ALL_SEEDS)} seeds per setup")
    print(f"{'='*110}")
    print(f"{'setup':<8}{'seeds':>7}{'Base':>9}{'B+KD':>9}{'Fair':>9}{'PQC_PE':>10}{'PQC_CE':>10}{'Q-RAD':>10}{'Δ_Base':>9}{'Δ_Fair':>9}")
    print("-" * 110)
    for tag in [s[0] for s in SETUPS]:
        res = all_results[tag]
        n = len(res)
        def mean(key, sub): return np.mean([r[key][sub] for r in res])
        base = mean("Baseline", "acc_all"); bkd = mean("Baseline_KD", "acc_all")
        fair = mean("Fair", "acc_all")
        pe = mean("PQC_PE", "acc_all"); ce = mean("PQC_CE_lam0p3", "acc_all")
        q  = mean("Q-RAD", "acc_all")
        print(f"{tag:<8}{n:>7}{base:>9.3f}{bkd:>9.3f}{fair:>9.3f}{pe:>10.3f}{ce:>10.3f}{q:>10.3f}"
              f"{q-base:>+9.4f}{q-fair:>+9.4f}")

    # Save aggregate
    (PAPER_DIR / "summary_all.json").write_text(json.dumps(all_results, indent=2, default=float))
    print(f"\n# saved {PAPER_DIR/'summary_all.json'}")

if __name__ == "__main__":
    main()
