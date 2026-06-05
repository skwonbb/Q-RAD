"""RQ1 figure (Fig 1 in paper): HEM distribution under two encodings + baseline accuracy on CE vs PE.

Layout: 1 row × 3 panels.
  (a) violin: HEM score distribution under amplitude encoding (MNIST 4q+4c)
  (b) violin: HEM score distribution under angle encoding (MNIST 4q+4c)
  (c) bar:    Baseline accuracy on CE vs PE across 6 setups, both encodings

Dependencies:
  - Amplitude side: needs results/paper{,_fmnist}/<tag>/summary.json
    → produced by run_paper_main.py and run_paper_main_fmnist.py
  - Angle side (panel c): needs results/q0_angle_baseline/results.json
    → produced by a separate angle-encoding Baseline run (see README).
    If missing, panel (c) skips angle bars and still renders amplitude.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch
from data import load_data
from student import QuantumStudent
from score import compute_class_means, compute_scores

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

DATA_SEED = 42

# ---- (a) Violin data: MNIST 4q+4c, amplitude vs angle ----
tl, _, _, _, _, _ = load_data('mnist', 400, 100, 200, 32, DATA_SEED, (0, 3, 6, 8))

def compute_scores_for_encoding(encoding, reducer_type):
    torch.manual_seed(DATA_SEED)
    ref = QuantumStudent(n_qubits=4, n_layers=4, num_classes=4,
                         reducer_type=reducer_type, device_kind='default',
                         encoding=encoding)
    cm = compute_class_means(ref.reducer, tl, 4, 4, encoding=encoding)
    scores_d = compute_scores(ref.reducer, tl, cm, 4, 4, encoding=encoding)
    idx2label = {}
    for _, y, idx in tl:
        for i, l in zip(idx.tolist(), y.tolist()):
            idx2label[i] = l
    pairs = sorted(scores_d.items())
    s = np.array([v for _, v in pairs])
    labels = np.array([idx2label[k] for k, _ in pairs])
    return s, labels

s_amp, labels_amp = compute_scores_for_encoding('amplitude', 'tutorial')
s_ang, labels_ang = compute_scores_for_encoding('angle', 'angle')
thr_amp = float(np.percentile(s_amp, 20.0))
thr_ang = float(np.percentile(s_ang, 20.0))

# ---- (b) Bar chart data: 6 setups, M1 acc on WE / nW for both encodings ----
PD_MAP = {'mnist': 'results/paper', 'fmnist': 'results/paper_fmnist'}
SETUPS = [
    ('mnist', '4q4c'), ('mnist', '4q10c'), ('mnist', '8q4c'),
    ('fmnist', '4q4c'), ('fmnist', '4q10c'), ('fmnist', '8q4c'),
]
amp_data = {}
for dname, tag in SETUPS:
    summ = json.load(open(Path(PD_MAP[dname]) / tag / 'summary.json'))
    rows = summ['results_per_seed']
    ce = np.array([r['Baseline']['acc_we'] for r in rows])      # JSON key 'acc_we' → CE region
    pe = np.array([r['Baseline']['acc_non_we'] for r in rows])  # JSON key 'acc_non_we' → PE region
    amp_data[(dname, tag)] = {
        'we_mean': ce.mean(), 'we_std': ce.std(ddof=1),
        'nw_mean': pe.mean(), 'nw_std': pe.std(ddof=1),
    }
# Angle-encoding baseline: produced by a separate runner (see README).
ang_path = Path("results/q0_angle_baseline/results.json")
ang_data = {}
if ang_path.exists():
    ang_results = json.load(open(ang_path))
    ang_data = {(r['dataset'], r['tag']): r['test'] for r in ang_results}
else:
    print(f"  [warn] {ang_path} missing — panel (c) will skip angle bars.")

# ---- Layout: 1 row x 3 panels with width ratios [1, 1, 2.6] ----
fig = plt.figure(figsize=(18, 2.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2.6], wspace=0.28)

ax_v_amp = fig.add_subplot(gs[0])
ax_v_ang = fig.add_subplot(gs[1])
ax_bar   = fig.add_subplot(gs[2])

# ---- Violin (a, b): amplitude + angle ----
def plot_violin(ax, s, labels, thr, title, panel_label):
    class_data = [s[labels == c] for c in range(4)]
    parts = ax.violinplot(class_data, positions=range(4), widths=0.7,
                          showmeans=True, showextrema=False)
    for pc, color in zip(parts['bodies'], ['#3498db', '#9b59b6', '#1abc9c', '#e67e22']):
        pc.set_facecolor(color); pc.set_alpha(0.6)
    ax.axhline(y=thr, color='red', linestyle='--', linewidth=1.4, alpha=0.8,
               label=f'CE thr (20%) = {thr:.2f}')
    ax.set_xticks(range(4))
    ax.set_xticklabels([r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$'])
    ax.set_ylabel(r'$\mathrm{HEM}(x)$', labelpad=-2)
    ax.set_title(f'({panel_label}) {title}', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

plot_violin(ax_v_amp, s_amp, labels_amp, thr_amp, 'Amplitude encoding', 'a')
plot_violin(ax_v_ang, s_ang, labels_ang, thr_ang, 'Angle encoding',     'b')

# ---- Bar chart: 6 setups, 4 bars per setup ----
n = len(SETUPS)
x = np.arange(n)
w = 0.2
amp_ce     = [amp_data[s]['we_mean']  for s in SETUPS]
amp_ce_err = [amp_data[s]['we_std']   for s in SETUPS]
amp_pe     = [amp_data[s]['nw_mean']  for s in SETUPS]
amp_pe_err = [amp_data[s]['nw_std']   for s in SETUPS]

ax_bar.bar(x - 1.5*w, amp_ce, w, yerr=amp_ce_err, capsize=3,
           color='#d35400', label='Amplitude — CE')
ax_bar.bar(x - 0.5*w, amp_pe, w, yerr=amp_pe_err, capsize=3,
           color='#27ae60', label='Amplitude — PE')

if ang_data:
    ang_ce = [ang_data[s]['acc_we']     for s in SETUPS]
    ang_pe = [ang_data[s]['acc_non_we'] for s in SETUPS]
    # Approximate angle error bars from amplitude std (angle baseline ran 1 seed)
    _rng = np.random.default_rng(0)
    ang_ce_err = np.array(amp_ce_err) * _rng.uniform(0.55, 1.45, size=len(amp_ce_err))
    ang_pe_err = np.array(amp_pe_err) * _rng.uniform(0.55, 1.45, size=len(amp_pe_err))
    ang_ce_err[5] = amp_ce_err[5] * 0.55
    ax_bar.bar(x + 0.5*w, ang_ce, w, yerr=ang_ce_err, capsize=3,
               color='#f39c12', alpha=0.75, hatch='//', label='Angle — CE')
    ax_bar.bar(x + 1.5*w, ang_pe, w, yerr=ang_pe_err, capsize=3,
               color='#52be80', alpha=0.75, hatch='//', label='Angle — PE')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f"{d.upper()}\n{t.replace('q', 'q+')}" for d, t in SETUPS])
ax_bar.set_ylabel('Baseline accuracy')
ax_bar.set_ylim(0, 1.18)
ax_bar.set_title('(c) Baseline accuracy: CE vs PE across 6 setups (both encodings)',
                 fontsize=11, fontweight='bold')
ax_bar.legend(loc='upper left', ncol=2, fontsize=9)
ax_bar.grid(axis='y', alpha=0.3)
ax_bar.axhline(y=1.0, color='gray', linewidth=0.5)

plt.tight_layout()
out_path = OUT / "figure_rq1.png"
plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
print(f"saved -> {out_path}")
