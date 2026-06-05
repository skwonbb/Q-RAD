"""Build the main paper table (Table 2): per-setup accuracy with π_neg.

Reads cached per-seed JSON outputs from run_paper_main{,_fmnist}.py and prints
mean ± std for Baseline, Baseline_KD, Fair, PQC_PE, PQC_CE_lam0p3 (and
PQC_CE_lam0p5 on 10-class setups), then computes oracle-routed Q-RAD.

Also computes π_neg := Pr[HEM(x) < 0] on the training set per setup.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path('.').resolve() / "src"))
from data import load_data
from student import QuantumStudent
from score import compute_class_means, compute_scores, make_we_mask, apply_threshold

DATA_SEED = 42

SETUPS = [
    # (display_ds, tag, dataset_key, n_qubits, num_classes, classes, results_dir, comp_ratio)
    ("MNIST",  "4q4c",  "mnist",  4,  4, (0, 3, 6, 8),     Path("results/paper/4q4c"),         49),
    ("MNIST",  "4q10c", "mnist",  4, 10, tuple(range(10)), Path("results/paper/4q10c"),        49),
    ("MNIST",  "8q4c",  "mnist",  8,  4, (0, 3, 6, 8),     Path("results/paper/8q4c"),          3),
    ("FMNIST", "4q4c",  "fmnist", 4,  4, (0, 1, 5, 9),     Path("results/paper_fmnist/4q4c"),  49),
    ("FMNIST", "4q10c", "fmnist", 4, 10, tuple(range(10)), Path("results/paper_fmnist/4q10c"), 49),
    ("FMNIST", "8q4c",  "fmnist", 8,  4, (0, 1, 5, 9),     Path("results/paper_fmnist/8q4c"),   3),
]


def stats(arr):
    arr = np.array(arr)
    return arr.mean(), arr.std(ddof=1)


def fmt(mean, std):
    return f"{mean*100:5.2f}$\\pm${std*100:4.2f}"


def route_all(p_pe, p_ce, ce_mask):
    """Net accuracy after oracle routing (CE → PQC_CE, PE → PQC_PE)."""
    n = c = 0
    for k in p_pe:
        v = p_ce[k] if int(k) in ce_mask else p_pe[k]
        n += 1; c += v["correct"]
    return c / n


def route_split(p_pe, p_ce, ce_mask):
    """Per-region accuracy after oracle routing."""
    n_ce = c_ce = n_pe = c_pe = 0
    for k in p_pe:
        in_ce = int(k) in ce_mask
        v = p_ce[k] if in_ce else p_pe[k]
        if in_ce: n_ce += 1; c_ce += v["correct"]
        else:     n_pe += 1; c_pe += v["correct"]
    return c_ce / n_ce, c_pe / n_pe


for ds, tag, dskey, nq, nc, cls, pd, comp in SETUPS:
    # Discover seeds from the cached seed_* directories
    seed_dirs = sorted(pd.glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    if not seed_dirs:
        print(f"\n[skip] {ds} {tag}: no seed_* directories under {pd}")
        continue
    SEEDS = [int(d.name.split("_")[1]) for d in seed_dirs]

    # π_neg from training scores
    tl, _, te, _, _, _ = load_data(dskey, 400, 100, 200, 32, DATA_SEED, cls)
    torch.manual_seed(DATA_SEED)
    ref = QuantumStudent(n_qubits=nq, n_layers=4, num_classes=nc,
                         reducer_type="tutorial", device_kind="default")
    cm = compute_class_means(ref.reducer, tl, nq, nc)
    tr_sc = compute_scores(ref.reducer, tl, cm, nq, nc)
    te_sc = compute_scores(ref.reducer, te, cm, nq, nc)
    pi_neg = (np.array(list(tr_sc.values())) < 0).mean()

    # CE mask on test (apply training-derived 20th-percentile threshold)
    _, thr = make_we_mask(tr_sc, percentile=20.0)
    test_ce = apply_threshold(te_sc, thr)

    print(f"\n## {ds} {tag}  (pi_neg={pi_neg*100:.1f}%, comp={comp}x, n_seeds={len(SEEDS)})")
    methods = ["Baseline", "Baseline_KD", "Fair", "PQC_PE", "PQC_CE_lam0p3"]
    if "10c" in tag:
        methods.append("PQC_CE_lam0p5")

    cache = {}  # per-method per-seed per_idx for oracle routing
    for m in methods:
        accs, ces, pes = [], [], []
        for s in SEEDS:
            d = json.load(open(pd / f"seed_{s}" / f"{m}.json"))
            accs.append(d["test"]["acc_all"])
            ces.append(d["test"]["acc_we"])     # JSON key from train.py is still acc_we
            pes.append(d["test"]["acc_non_we"]) # JSON key from train.py is still acc_non_we
            cache.setdefault(m, {})[s] = d.get("per_idx")
        am, asd = stats(accs); cm_, csd = stats(ces); pm, psd = stats(pes)
        print(f"  {m:<16} all={fmt(am,asd)}  CE={fmt(cm_,csd)}  PE={fmt(pm,psd)}")

    # Q-RAD with PQC_CE_lam0p3
    oas, ocs, ops = [], [], []
    for s in SEEDS:
        p_pe = cache["PQC_PE"][s]; p_ce = cache["PQC_CE_lam0p3"][s]
        oas.append(route_all(p_pe, p_ce, test_ce))
        oc, op = route_split(p_pe, p_ce, test_ce)
        ocs.append(oc); ops.append(op)
    am, asd = stats(oas); cm_, csd = stats(ocs); pm, psd = stats(ops)
    print(f"  Q-RAD(λ=0.3)     all={fmt(am,asd)}  CE={fmt(cm_,csd)}  PE={fmt(pm,psd)}")

    if "10c" in tag:
        oas, ocs, ops = [], [], []
        for s in SEEDS:
            p_pe = cache["PQC_PE"][s]; p_ce = cache["PQC_CE_lam0p5"][s]
            oas.append(route_all(p_pe, p_ce, test_ce))
            oc, op = route_split(p_pe, p_ce, test_ce)
            ocs.append(oc); ops.append(op)
        am, asd = stats(oas); cm_, csd = stats(ocs); pm, psd = stats(ops)
        print(f"  Q-RAD(λ=0.5)     all={fmt(am,asd)}  CE={fmt(cm_,csd)}  PE={fmt(pm,psd)}")
