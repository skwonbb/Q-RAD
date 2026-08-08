# Q-RAD (CIKM 2026)

**Q-RAD** is a dual-PQC knowledge-distillation framework that trains two compact
specialists - one for the Compression-affected Encoding (CE) region and one for
the Properly Encoded (PE) region - and combines them via oracle routing.
Experiments cover MNIST and FashionMNIST under three (qubit, class) configurations.

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.10+ recommended. MNIST and FashionMNIST are downloaded automatically by
`torchvision` on first run into `./data/`.

## Reproducing each Research Question

All commands run from the repository root. Outputs are written under `results/`.

### RQ1 — CE/PE partition validity (Figure 1)
```bash
python run_paper_main.py          # amplitude Baselines on MNIST (also used by RQ2-5)
python run_paper_main_fmnist.py   # amplitude Baselines on FMNIST
python run_angle_baseline.py      # angle-encoding Baselines, 5 seeds × 6 setups (MNIST + FMNIST)
python figure_rq1.py              # writes results/figures/figure_rq1.png
```

### RQ2 / RQ3 — Specialist accuracy and architectural separation (Table 2)
```bash
python run_paper_main.py          # MNIST: Baseline, Baseline_KD, Fair, PQC_PE, PQC_CE_*
python run_paper_main_fmnist.py   # FMNIST: same five models
python build_main_table.py        # prints Table 2 (mean ± std across 5 seeds)
```

### RQ4 — HEM and encoding compression (Table 2 π_neg column + footnote)
```bash
python build_main_table.py        # π_neg column for the 6 main setups
python compute_pi_neg_8q10c.py    # footnote: 8q+10c on MNIST and FMNIST
```

### RQ5 — Hilbert-space vs pixel-space CE selection (Figure 2 + Table 3)
```bash
# Hilbert-side specialists come from run_paper_main{,_fmnist}.py above.
python run_pixel_specialist.py    # pixel-space CE specialists, 5 seeds × 6 setups
python build_pixel_table.py       # prints Table 3
python figure_rq5_disagreement.py # writes results/figures/figure_rq5.png
```

A full reproduction (all five RQs) takes several hours of wall time on a CPU.
Per-method training is single-process; running multiple methods/setups in
parallel is left to the user.

## Method naming

The paper uses descriptive labels; the code writes per-seed JSON outputs
under the same names. Inside `src/train.py`, `acc_we` / `acc_non_we` in the
evaluator output correspond to **CE** / **PE** region accuracy, respectively.

| Paper            | Output file                | `method_id` | Description                                  |
|------------------|----------------------------|:-:|----------------------------------------------|
| Baseline         | `Baseline.json`            | 1 | 4-layer PQC, no KD                           |
| Baseline+KD      | `Baseline_KD.json`         | 3 | 4-layer PQC, uniform KD (λ=0.1, T=2)         |
| Fair             | `Fair.json`                | 1 | 8-layer PQC, no KD (parameter-matched)       |
| PQC_PE           | `PQC_PE.json`              | 6 | 4-layer specialist, KD on PE only (λ=0.1)    |
| PQC_CE           | `PQC_CE_lam0p3.json`       | 8 | 4-layer specialist, KD on CE only (λ=0.3)    |
| PQC_CE (10-class)| `PQC_CE_lam0p5.json`       | 8 | 4-layer specialist, KD on CE only (λ=0.5)    |
| Q-RAD            | summary key `Q-RAD`        | — | oracle routing of PQC_PE and PQC_CE         |

## Statistical Reproducibility

Reported results are averaged over 5 random seeds (mean ± std). Each
training script draws 5 fresh seeds at startup; analysis scripts auto-discover
the seeds from `seed_*/` directories under `results/`, so the workflow does
not require pinning specific seed values. Run the training script with any
seeds to reproduce statistical trends.

A fixed `DATA_SEED = 42` controls which samples are drawn from MNIST and
FashionMNIST for the train/val/test subsets so that evaluation samples are
consistent across runs.

## Repository structure

```
.
├── src/
│   ├── data.py        # MNIST/FashionMNIST loader, 4- or 10-class subsets
│   ├── student.py     # Tutorial / Angle reducers + QuantumStudent (PQC)
│   ├── teacher.py     # LeNet teacher + training loop
│   ├── score.py       # HEM (Hilbert-space encoding margin) + CE mask
│   └── train.py       # train_method (KD variants) + evaluate
│
├── run_paper_main.py            # RQ2/3 main training, MNIST
├── run_paper_main_fmnist.py     # RQ2/3 main training, FashionMNIST
├── run_angle_baseline.py        # RQ1 angle-encoding Baseline
├── run_pixel_specialist.py      # RQ5 pixel-space CE specialists
│
├── build_main_table.py          # Table 2 + π_neg
├── build_pixel_table.py         # Table 3
├── compute_pi_neg_8q10c.py      # RQ4 footnote (π_neg at 8q+10c)
│
├── figure_rq1.py                # Figure 1
└── figure_rq5_disagreement.py   # Figure 2
```
