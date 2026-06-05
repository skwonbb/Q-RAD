"""Build the RQ5 ablation table (Table 3): Hilbert (HEM) vs pixel CE-selection.

Reads cached pixel specialists from results/pixel_specialist/<setup>/
and the Hilbert specialists from results/paper{,_fmnist}/<tag>/seed_<seed>/.

Outputs per-setup PQC_CE accuracy split by region (CE / PE) for both
selection variants, and the Q-RAD net accuracy after oracle routing.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path('.').resolve() / "src"))
from data import load_data
from student import QuantumStudent
from score import compute_class_means, compute_scores, make_we_mask, apply_threshold

DATA_SEED = 42
SETUPS = [
    # (ds_key, display_label, nq, nc, classes, hilbert_results_dir, hilbert_ce_label)
    ("mnist",  "MNIST_4q4c",   4,  4, (0, 3, 6, 8),     Path("results/paper/4q4c"),         "PQC_CE_lam0p3"),
    ("mnist",  "MNIST_4q10c",  4, 10, tuple(range(10)), Path("results/paper/4q10c"),        "PQC_CE_lam0p5"),
    ("mnist",  "MNIST_8q4c",   8,  4, (0, 3, 6, 8),     Path("results/paper/8q4c"),         "PQC_CE_lam0p3"),
    ("fmnist", "FMNIST_4q4c",  4,  4, (0, 1, 5, 9),     Path("results/paper_fmnist/4q4c"),  "PQC_CE_lam0p3"),
    ("fmnist", "FMNIST_4q10c", 4, 10, tuple(range(10)), Path("results/paper_fmnist/4q10c"), "PQC_CE_lam0p5"),
    ("fmnist", "FMNIST_8q4c",  8,  4, (0, 1, 5, 9),     Path("results/paper_fmnist/8q4c"),  "PQC_CE_lam0p3"),
]


def pooled(reducer, loader, nq):
    out = {}; reducer.eval()
    with torch.no_grad():
        for x, y, idx in loader:
            v = reducer(x).cpu().numpy()
            for i in range(v.shape[0]):
                out[int(idx[i])] = (v[i], int(y[i]))
    return out


def s_pixel(v_dict, cm):
    out = {}; norms_c = np.linalg.norm(cm, axis=1)
    for idx, (vi, yi) in v_dict.items():
        n = np.linalg.norm(vi) + 1e-12
        sims = (cm @ vi) / (norms_c * n + 1e-12)
        own = sims[yi]; others = np.delete(sims, yi)
        out[idx] = float(own - others.max())
    return out


def route(p_pe, p_ce, ce_mask):
    n = c = 0
    for k in p_pe:
        v = p_ce[k] if int(k) in ce_mask else p_pe[k]
        n += 1; c += v["correct"]
    return c / n if n else 0


def split_acc(p, ce_mask):
    """Return (acc_all, acc_ce, acc_pe) for one model's per_idx dict."""
    n_a = c_a = n_ce = c_ce = n_pe = c_pe = 0
    for k, v in p.items():
        n_a += 1; c_a += v["correct"]
        if int(k) in ce_mask: n_ce += 1; c_ce += v["correct"]
        else:                 n_pe += 1; c_pe += v["correct"]
    return c_a / n_a, (c_ce / n_ce if n_ce else 0), (c_pe / n_pe if n_pe else 0)


PSP = Path("results/pixel_specialist")
for ds_key, label, nq, nc, cls, pd_h, h_ce_label in SETUPS:
    setup_dir = PSP / label
    if not setup_dir.exists():
        print(f"\n[skip] {label}: no pixel-specialist outputs (run run_pixel_specialist.py first).")
        continue
    tl, _, te, _, _, _ = load_data(ds_key, 400, 100, 200, 32, DATA_SEED, cls)
    torch.manual_seed(DATA_SEED)
    ref = QuantumStudent(n_qubits=nq, n_layers=4, num_classes=nc,
                         reducer_type="tutorial", device_kind="default")
    # Pixel-CE mask
    tr_v = pooled(ref.reducer, tl, nq); te_v = pooled(ref.reducer, te, nq)
    cm = np.zeros((nc, 2**nq)); cnt = np.zeros(nc)
    for vi, yi in tr_v.values():
        cm[yi] += vi; cnt[yi] += 1
    cm /= cnt[:, None]
    tr_p = s_pixel(tr_v, cm); te_p = s_pixel(te_v, cm)
    thr_p = np.percentile(np.array(list(tr_p.values())), 20.0)
    test_ce_pixel = set(i for i, s in te_p.items() if s <= thr_p)
    # Hilbert-CE mask
    cm_h = compute_class_means(ref.reducer, tl, nq, nc)
    tr_h = compute_scores(ref.reducer, tl, cm_h, nq, nc)
    te_h = compute_scores(ref.reducer, te, cm_h, nq, nc)
    _, thr_h = make_we_mask(tr_h, percentile=20.0)
    test_ce_hilbert = apply_threshold(te_h, thr_h)

    # Discover seeds present in both pixel-specialist and Hilbert specialist directories
    pixel_seeds = {int(p.stem.rsplit("_", 1)[1]) for p in setup_dir.glob("PQC_PE_pixel_seed_*.json")}
    hilbert_seeds = {int(d.name.split("_")[1]) for d in pd_h.glob("seed_*")}
    seeds = sorted(pixel_seeds & hilbert_seeds)

    print(f"\n=== {label} ===  (n_seeds={len(seeds)})")
    rows = {"PQC_PE_H": [], "PQC_PE_P": [], "PQC_CE_H": [], "PQC_CE_P": [],
             "QRAD_H": [], "QRAD_P": []}
    for seed in seeds:
        pe_p = setup_dir / f"PQC_PE_pixel_seed_{seed}.json"
        ce_p = setup_dir / f"PQC_CE_pixel_seed_{seed}.json"
        if not (pe_p.exists() and ce_p.exists()):
            continue
        d_pe_pix = {int(k): v for k, v in json.load(open(pe_p))["per_idx"].items()}
        d_ce_pix = {int(k): v for k, v in json.load(open(ce_p))["per_idx"].items()}

        seed_dir = pd_h / f"seed_{seed}"
        d_pe_h = {int(k): v for k, v in json.load(open(seed_dir / "PQC_PE.json"))["per_idx"].items()}
        d_ce_h = {int(k): v for k, v in json.load(open(seed_dir / f"{h_ce_label}.json"))["per_idx"].items()}

        # Pixel side: split by pixel-CE; Hilbert side: split by Hilbert-CE
        rows["PQC_PE_P"].append(split_acc(d_pe_pix, test_ce_pixel))
        rows["PQC_CE_P"].append(split_acc(d_ce_pix, test_ce_pixel))
        rows["QRAD_P"].append(route(d_pe_pix, d_ce_pix, test_ce_pixel))
        rows["PQC_PE_H"].append(split_acc(d_pe_h, test_ce_hilbert))
        rows["PQC_CE_H"].append(split_acc(d_ce_h, test_ce_hilbert))
        rows["QRAD_H"].append(route(d_pe_h, d_ce_h, test_ce_hilbert))

    n_done = len(rows["QRAD_H"])
    if n_done == 0:
        print(f"  no seeds with both Hilbert and pixel specialists cached.")
        continue
    print(f"  ({n_done} seeds completed)")
    print(f"  {'model':<14}{'all':>9}{'CE':>9}{'PE':>9}")
    for key in ["PQC_PE_H", "PQC_PE_P", "PQC_CE_H", "PQC_CE_P"]:
        a = np.array(rows[key])
        m = a.mean(axis=0)
        print(f"  {key:<14}{m[0]:>9.3f}{m[1]:>9.3f}{m[2]:>9.3f}")
    qh = np.mean(rows["QRAD_H"]); qp = np.mean(rows["QRAD_P"])
    print(f"  {'Q-RAD_H':<14}{qh:>9.3f}")
    print(f"  {'Q-RAD_P':<14}{qp:>9.3f}    Δ(P-H) = {qp-qh:+.4f}")
