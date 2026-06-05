"""Figure: pixel-score vs quantum-score scatter, with threshold lines and quadrant coloring.
Shows that the two scores disagree on which samples are CE — and the disagreement region
(p-only, q-only) is where the experimental gap lives.

Layout: 1 row (FMNIST only) x 3 cols (4q+4c, 4q+10c, 8q+4c).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data import load_data
from student import QuantumStudent
from score import compute_class_means, compute_scores, make_we_mask, apply_threshold

DATA_SEED = 42
SETUPS = [
    ("fmnist", "4-Qubit 4-Class",  4,  4, (0,1,5,9)),
    ("fmnist", "4-Qubit 10-Class", 4, 10, tuple(range(10))),
    ("fmnist", "8-Qubit 4-Class",  8,  4, (0,1,5,9)),
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

# Quadrant colors
COL_BOTH    = "#7c3aed"  # purple
COL_Q_ONLY  = "#3b6fb1"  # blue
COL_P_ONLY  = "#d96a47"  # orange
COL_NEITHER = "#cccccc"  # light gray

plt.rcParams.update({
    "font.size": 14,
    "axes.titleweight": "bold",   # only titles bold
    "axes.titlesize": 16,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
})

# FMNIST only: 1 row x 3 cols
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

for c_i, (ds_key, label, nq, nc, cls) in enumerate(SETUPS):
    tl, _, te, _, _, _ = load_data(ds_key, 400, 100, 200, 32, DATA_SEED, cls)
    torch.manual_seed(DATA_SEED)
    ref = QuantumStudent(n_qubits=nq, n_layers=4, num_classes=nc,
                         reducer_type="tutorial", device_kind="default")
    # quantum score
    cm_q = compute_class_means(ref.reducer, tl, nq, nc)
    tr_q = compute_scores(ref.reducer, tl, cm_q, nq, nc)
    te_q = compute_scores(ref.reducer, te, cm_q, nq, nc)
    _, thr_q = make_we_mask(tr_q, percentile=20.0)
    # pixel score
    tr_v = pooled(ref.reducer, tl, nq); te_v = pooled(ref.reducer, te, nq)
    cm = np.zeros((nc, 2**nq)); cnt = np.zeros(nc)
    for vi, yi in tr_v.values():
        cm[yi] += vi; cnt[yi] += 1
    cm /= cnt[:, None]
    tr_p = s_pixel(tr_v, cm); te_p = s_pixel(te_v, cm)
    thr_p = np.percentile(np.array(list(tr_p.values())), 20.0)

    # Collect test samples
    idxs = sorted(te_q.keys())
    xs = np.array([te_q[i] for i in idxs])  # quantum
    ys = np.array([te_p[i] for i in idxs])  # pixel
    in_q = xs <= thr_q
    in_p = ys <= thr_p

    # Quadrant masks
    both    = in_q & in_p
    q_only  = in_q & (~in_p)
    p_only  = (~in_q) & in_p
    neither = (~in_q) & (~in_p)

    ax = axes[c_i]
    # Faint neither (background only)
    ax.scatter(xs[neither], ys[neither], s=4, c=COL_NEITHER, alpha=0.18, label="neither")
    # Colored quadrants — larger markers so visible at small render
    ax.scatter(xs[both],    ys[both],    s=22, c=COL_BOTH,    alpha=0.9, label="both")
    ax.scatter(xs[q_only],  ys[q_only],  s=22, c=COL_Q_ONLY,  alpha=0.9, label="only by HEM")
    ax.scatter(xs[p_only],  ys[p_only],  s=22, c=COL_P_ONLY,  alpha=0.9, label="only by NCM")

    # threshold lines
    ax.axvline(thr_q, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(thr_p, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

    # Subplot title — bold via rcParams default
    ax.set_title(label)
    # tick density: reduce to ~3 per axis so they don't overlap at large font
    ax.locator_params(axis='both', nbins=4)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

# Single shared legend BELOW the figure (outside the plot area, so subplots keep full size)
handles = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=COL_NEITHER, markersize=18, label='neither'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=COL_BOTH,    markersize=18, label='both'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=COL_Q_ONLY,  markersize=18, label='only by HEM'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=COL_P_ONLY,  markersize=18, label='only by NCM'),
]
plt.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6, rect=[0, 0.12, 1, 1])
fig.legend(handles=handles, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.02), frameon=False)
out = Path("results/figures/figure_rq5.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"saved -> {out}")
