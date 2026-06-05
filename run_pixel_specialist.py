"""RQ5: train pixel-space CE specialists and compare Q-RAD against the Hilbert (HEM) version.

For each setup, trains PQC_PE_pixel (KD on PE-pixel, method_id=6) and
PQC_CE_pixel (KD on CE-pixel, method_id=8) where the CE partition uses
a nearest-class-mean cosine margin on the pooled pixel vector instead of
the post-amplitude-embedding HEM score. Test routing uses the same
pixel-CE membership.

Outputs to results/pixel_specialist/<setup>/{PQC_PE_pixel_seed_S,PQC_CE_pixel_seed_S}.json.
Quantum-side specialists (PQC_PE, PQC_CE_lam0p3 / PQC_CE_lam0p5) are read
from the cached run_paper_main outputs.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data import load_data
from teacher import TeacherCNN
from student import QuantumStudent
from train import train_method, evaluate

DATA_SEED = 42       # fixed: controls which samples are drawn for train/val/test
# Training seeds discovered per-setup from the cached Hilbert-side results so
# pixel and Hilbert specialists are trained with the same seeds for apples-to-apples.
N_EPOCHS = 50; LR = 1e-3
PERCENTILE = 20.0
T = 2.0
PQC_PE_LAM = 0.1

SETUPS = [
    # (ds_key, label, nq, nc, cls, results_dir, pqc_ce_lambda, quantum_ce_label)
    ("mnist",  "MNIST_4q4c",   4,  4, (0, 3, 6, 8),     Path("results/paper/4q4c"),         0.3, "PQC_CE_lam0p3"),
    ("mnist",  "MNIST_4q10c",  4, 10, tuple(range(10)), Path("results/paper/4q10c"),        0.5, "PQC_CE_lam0p5"),
    ("mnist",  "MNIST_8q4c",   8,  4, (0, 3, 6, 8),     Path("results/paper/8q4c"),         0.3, "PQC_CE_lam0p3"),
    ("fmnist", "FMNIST_4q4c",  4,  4, (0, 1, 5, 9),     Path("results/paper_fmnist/4q4c"),  0.3, "PQC_CE_lam0p3"),
    ("fmnist", "FMNIST_4q10c", 4, 10, tuple(range(10)), Path("results/paper_fmnist/4q10c"), 0.5, "PQC_CE_lam0p5"),
    ("fmnist", "FMNIST_8q4c",  8,  4, (0, 1, 5, 9),     Path("results/paper_fmnist/8q4c"),  0.3, "PQC_CE_lam0p3"),
]


def pooled_vectors(reducer, loader, nq):
    out = {}; reducer.eval()
    with torch.no_grad():
        for x, y, idx in loader:
            v = reducer(x).cpu().numpy()
            for i in range(v.shape[0]):
                out[int(idx[i])] = (v[i], int(y[i]))
    return out


def s_pixel_cos(v_dict, class_means):
    """Nearest-class-mean cosine margin in pixel space."""
    out = {}
    norms_c = np.linalg.norm(class_means, axis=1)
    for idx, (vi, yi) in v_dict.items():
        n = np.linalg.norm(vi) + 1e-12
        sims = (class_means @ vi) / (norms_c * n + 1e-12)
        own = sims[yi]; others = np.delete(sims, yi)
        out[idx] = float(own - others.max())
    return out


def per_sample(model, loader):
    out = {}; model.eval()
    with torch.no_grad():
        for x, y, idx in loader:
            logits = model(x); preds = logits.argmax(dim=-1)
            for i in range(x.size(0)):
                out[int(idx[i])] = {"pred": int(preds[i]),
                                    "correct": int(preds[i] == y[i])}
    return out


def route_acc(p_pe, p_ce, ce_mask):
    n = c = 0
    for k in p_pe:
        v = p_ce[k] if int(k) in ce_mask else p_pe[k]
        n += 1; c += v["correct"]
    return c / n if n else 0.0


OUT = Path("results/pixel_specialist")
OUT.mkdir(parents=True, exist_ok=True)

summary = []
for ds_key, ds_label, nq, nc, cls, pd_orig, pqc_ce_lam, q_ce_label in SETUPS:
    seeds = sorted(int(d.name.split("_")[1]) for d in pd_orig.glob("seed_*"))
    if not seeds:
        print(f"\n[skip] {ds_label}: no seeds under {pd_orig} — run run_paper_main first.")
        continue
    print(f"\n========== {ds_label}  PQC_CE λ={pqc_ce_lam}  ({len(seeds)} seeds: {seeds}) ==========")
    tl, vl, te, train_ds, _, _ = load_data(ds_key, 400, 100, 200, 32, DATA_SEED, cls)
    teacher = TeacherCNN(num_classes=nc)
    teacher.load_state_dict(torch.load(pd_orig / "teacher.pt", map_location="cpu"))
    teacher.eval()

    # Pixel-space scores + 20th-percentile CE threshold
    torch.manual_seed(DATA_SEED)
    ref = QuantumStudent(n_qubits=nq, n_layers=4, num_classes=nc,
                         reducer_type="tutorial", device_kind="default")
    tr_v = pooled_vectors(ref.reducer, tl, nq)
    vl_v = pooled_vectors(ref.reducer, vl, nq)
    te_v = pooled_vectors(ref.reducer, te, nq)
    cmp = np.zeros((nc, 2**nq)); cnt = np.zeros(nc)
    for vi, yi in tr_v.values():
        cmp[yi] += vi; cnt[yi] += 1
    cmp /= cnt[:, None]
    tr_p = s_pixel_cos(tr_v, cmp)
    vl_p = s_pixel_cos(vl_v, cmp)
    te_p = s_pixel_cos(te_v, cmp)
    thr = np.percentile(np.array(list(tr_p.values())), PERCENTILE)
    train_ce_p = set(idx for idx, s in tr_p.items() if s <= thr)
    val_ce_p   = set(idx for idx, s in vl_p.items() if s <= thr)
    test_ce_p  = set(idx for idx, s in te_p.items() if s <= thr)
    print(f"  pixel-CE: train={len(train_ce_p)}, val={len(val_ce_p)}, test={len(test_ce_p)}")

    setup_dir = OUT / ds_label; setup_dir.mkdir(parents=True, exist_ok=True)

    qrad_hilbert = []; qrad_pixel = []
    for seed in seeds:
        # ---- PQC_PE_pixel ----
        pe_json = setup_dir / f"PQC_PE_pixel_seed_{seed}.json"
        if pe_json.exists():
            d_pe = json.loads(pe_json.read_text())
            print(f"  [seed {seed}] PQC_PE_pixel cache hit")
        else:
            g = torch.Generator().manual_seed(seed)
            loader = DataLoader(Subset(train_ds, list(range(len(train_ds)))),
                                batch_size=32, shuffle=True, generator=g)
            torch.manual_seed(seed)
            s_pe = QuantumStudent(n_qubits=nq, n_layers=4, num_classes=nc,
                                reducer_type="tutorial", device_kind="default")
            t0 = time.perf_counter()
            train_method(method_id=6, student=s_pe, teacher=teacher,
                         train_loader=loader, val_loader=vl,
                         train_we_mask=train_ce_p, val_we_mask=val_ce_p,
                         n_epochs=N_EPOCHS, lr=LR, log_every=N_EPOCHS,
                         lambda_kd=PQC_PE_LAM, temperature=T)
            el = time.perf_counter() - t0
            s_pe.eval()
            d_pe = {"per_idx": per_sample(s_pe, te), "elapsed": el}
            pe_json.write_text(json.dumps(d_pe, indent=2, default=float))
            print(f"  [seed {seed}] PQC_PE_pixel trained ({el:.0f}s)")

        # ---- PQC_CE_pixel ----
        ce_json = setup_dir / f"PQC_CE_pixel_seed_{seed}.json"
        if ce_json.exists():
            d_ce = json.loads(ce_json.read_text())
            print(f"  [seed {seed}] PQC_CE_pixel cache hit")
        else:
            g = torch.Generator().manual_seed(seed)
            loader = DataLoader(Subset(train_ds, list(range(len(train_ds)))),
                                batch_size=32, shuffle=True, generator=g)
            torch.manual_seed(seed)
            s_ce = QuantumStudent(n_qubits=nq, n_layers=4, num_classes=nc,
                                reducer_type="tutorial", device_kind="default")
            t0 = time.perf_counter()
            train_method(method_id=8, student=s_ce, teacher=teacher,
                         train_loader=loader, val_loader=vl,
                         train_we_mask=train_ce_p, val_we_mask=val_ce_p,
                         n_epochs=N_EPOCHS, lr=LR, log_every=N_EPOCHS,
                         lambda_kd=pqc_ce_lam, temperature=T)
            el = time.perf_counter() - t0
            s_ce.eval()
            d_ce = {"per_idx": per_sample(s_ce, te), "elapsed": el}
            ce_json.write_text(json.dumps(d_ce, indent=2, default=float))
            print(f"  [seed {seed}] PQC_CE_pixel trained ({el:.0f}s)")

        # ---- Pixel-side Q-RAD: route by pixel-CE ----
        p_pe_pix = {int(k): v for k, v in d_pe["per_idx"].items()}
        p_ce_pix = {int(k): v for k, v in d_ce["per_idx"].items()}
        qrad_p = route_acc(p_pe_pix, p_ce_pix, test_ce_p)

        # ---- Hilbert-side Q-RAD: route by HEM-CE (recompute mask on the fly) ----
        from score import compute_class_means, compute_scores, make_we_mask, apply_threshold
        torch.manual_seed(DATA_SEED)
        ref_h = QuantumStudent(n_qubits=nq, n_layers=4, num_classes=nc,
                               reducer_type="tutorial", device_kind="default")
        cm_h = compute_class_means(ref_h.reducer, tl, nq, nc)
        tr_h = compute_scores(ref_h.reducer, tl, cm_h, nq, nc)
        te_h = compute_scores(ref_h.reducer, te, cm_h, nq, nc)
        _, thr_h = make_we_mask(tr_h, percentile=PERCENTILE)
        test_ce_h = apply_threshold(te_h, thr_h)

        p_pe_h = {int(k): v for k, v in
                  json.load(open(pd_orig / f"seed_{seed}" / "PQC_PE.json"))["per_idx"].items()}
        p_ce_h = {int(k): v for k, v in
                  json.load(open(pd_orig / f"seed_{seed}" / f"{q_ce_label}.json"))["per_idx"].items()}
        qrad_h = route_acc(p_pe_h, p_ce_h, test_ce_h)

        qrad_hilbert.append(qrad_h); qrad_pixel.append(qrad_p)
        print(f"  [seed {seed}] Q-RAD(HEM)={qrad_h:.3f}  Q-RAD(pixel)={qrad_p:.3f}  Δ={qrad_p-qrad_h:+.4f}")

    h_m, h_s = np.mean(qrad_hilbert), np.std(qrad_hilbert, ddof=1)
    p_m, p_s = np.mean(qrad_pixel),   np.std(qrad_pixel,   ddof=1)
    diff = p_m - h_m
    summary.append((ds_label, h_m, h_s, p_m, p_s, diff))
    print(f"  *** {ds_label} ***  HEM: {h_m:.3f}±{h_s:.3f}   pixel: {p_m:.3f}±{p_s:.3f}   Δ={diff:+.4f}")

print(f"\n\n========== SUMMARY ==========")
print(f"{'Setup':<14}{'Q-RAD (HEM)':>22}{'Q-RAD (pixel)':>20}{'Δ (p-H)':>11}")
print("-" * 70)
for label, h_m, h_s, p_m, p_s, d in summary:
    print(f"{label:<14}{h_m*100:>16.2f}±{h_s*100:.2f}{p_m*100:>16.2f}±{p_s*100:.2f}{d*100:>10.2f}")
