"""Encoding score (paper §3.2): per-sample class-discriminability after
amplitude encoding.

Two-stage API:
  1) `compute_class_means(reducer, train_loader, ...)` builds the per-class
     mean density matrices once on the training split.
  2) `compute_scores(reducer, loader, class_means, ...)` scores any split
     (train / val / test) against those *training-derived* means.

This split is what the toy guide §4.1 originally lacked — without it, the val
and test "WE-region" labels would be defined against their own means, drifting
from the train-set definition.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _statevector_amplitude(reduced_x: np.ndarray, n_qubits: int) -> np.ndarray:
    """Mirror `qml.AmplitudeEmbedding(..., normalize=True)`: L2-normalize the
    real reduced vector and treat it as a statevector amplitude (real-valued).

    Returns a complex array of length 2**n_qubits. Zero-norm inputs map to
    |0...0⟩ as a defined fallback.
    """
    norm = float(np.linalg.norm(reduced_x))
    if norm < 1e-10:
        sv = np.zeros(2 ** n_qubits, dtype=np.complex128)
        sv[0] = 1.0
        return sv
    return (reduced_x.astype(np.float64) / norm).astype(np.complex128)


def _statevector_angle(angles: np.ndarray, n_qubits: int) -> np.ndarray:
    """Mirror `qml.AngleEmbedding(..., rotation='Y')`: per-qubit R_Y(θ_i)|0⟩,
    giving a product state $\\bigotimes_i (\\cos(θ_i/2)|0⟩ + \\sin(θ_i/2)|1⟩)$.

    Builds the 2^n complex statevector via repeated Kronecker product.
    """
    if len(angles) != n_qubits:
        raise ValueError(f"angles length {len(angles)} != n_qubits {n_qubits}")
    state = np.array([1.0 + 0j], dtype=np.complex128)
    for theta in angles:
        single = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)], dtype=np.complex128)
        state = np.kron(state, single)
    return state


def _encode_batch(reducer: torch.nn.Module, loader, n_qubits: int, device: str,
                   encoding: str = "amplitude"):
    """Iterate `loader`, push each batch through `reducer`, and yield
    (sample_idx, label, statevector) for every sample.

    `encoding` selects how reducer outputs map to a 2^n statevector:
      - 'amplitude': reducer outputs 2^n real values, L2-normalized into amplitudes
      - 'angle': reducer outputs n_qubits angles, each becomes R_Y rotation on one qubit
    """
    reducer.eval()
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            reduced = reducer(x).cpu().numpy()
            for i in range(reduced.shape[0]):
                if encoding == "amplitude":
                    sv = _statevector_amplitude(reduced[i], n_qubits)
                elif encoding == "angle":
                    sv = _statevector_angle(reduced[i], n_qubits)
                else:
                    raise ValueError(f"Unknown encoding: {encoding}")
                yield int(idx[i]), int(y[i]), sv


def compute_class_means(
    reducer: torch.nn.Module,
    train_loader,
    n_qubits: int,
    num_classes: int,
    device: str = "cpu",
    encoding: str = "amplitude",
) -> list[np.ndarray]:
    """Per-class average density matrix ρ̄_c = (1/|S_c|) Σ |ψ(x)⟩⟨ψ(x)|.

    Returns a list of complex density matrices, indexed by remapped class id
    (0 .. num_classes-1). Shape of each: (2**n_qubits, 2**n_qubits).
    """
    dim = 2 ** n_qubits
    sums = [np.zeros((dim, dim), dtype=np.complex128) for _ in range(num_classes)]
    counts = [0] * num_classes
    for _, label, sv in _encode_batch(reducer, train_loader, n_qubits, device, encoding=encoding):
        sums[label] += np.outer(sv, sv.conj())
        counts[label] += 1
    means: list[np.ndarray] = []
    for c in range(num_classes):
        if counts[c] == 0:
            raise RuntimeError(f"No training samples found for class {c}.")
        means.append(sums[c] / counts[c])
    return means


def compute_scores(
    reducer: torch.nn.Module,
    loader,
    class_means: list[np.ndarray],
    n_qubits: int,
    num_classes: int,
    device: str = "cpu",
    encoding: str = "amplitude",
) -> dict[int, float]:
    """For each sample (x, y, idx) in `loader`, compute
        s(x) = ⟨ψ(x)| ρ̄_y |ψ(x)⟩ − max_{c≠y} ⟨ψ(x)| ρ̄_c |ψ(x)⟩
    using the supplied (typically training-derived) `class_means`.
    """
    if len(class_means) != num_classes:
        raise ValueError("len(class_means) must equal num_classes.")
    scores: dict[int, float] = {}
    for idx, label, sv in _encode_batch(reducer, loader, n_qubits, device, encoding=encoding):
        # ⟨ψ| ρ |ψ⟩ for each class — real by construction (ρ Hermitian)
        fids = [float(np.real(sv.conj() @ rho @ sv)) for rho in class_means]
        same = fids[label]
        other_max = max(f for c, f in enumerate(fids) if c != label)
        scores[idx] = same - other_max
    return scores


def make_we_mask(
    scores: dict[int, float],
    percentile: float = 20.0,
) -> tuple[set[int], float]:
    """Bottom-`percentile`% of scores are tagged WE.

    Returns (set_of_we_indices, threshold). The threshold is the percentile
    cutoff itself — store it so val/test can be classified against the *same*
    train-derived cutoff.
    """
    if not scores:
        return set(), float("nan")
    arr = np.array(list(scores.values()), dtype=np.float64)
    threshold = float(np.percentile(arr, percentile))
    we = {idx for idx, s in scores.items() if s < threshold}
    return we, threshold


def apply_threshold(scores: dict[int, float], threshold: float) -> set[int]:
    """Apply a precomputed threshold (from train) to a val/test score dict."""
    return {idx for idx, s in scores.items() if s < threshold}


if __name__ == "__main__":
    # Sanity check: small dry run on real data.
    import time
    from data import load_data
    from student import TutorialReducer

    torch.manual_seed(42)
    tl, vl, te, tds, vds, teds = load_data("mnist", batch_size=32, seed=42)

    reducer = TutorialReducer(n_qubits=4)

    t0 = time.perf_counter()
    class_means = compute_class_means(reducer, tl, n_qubits=4, num_classes=4)
    t_means = time.perf_counter() - t0
    print(f"class_means: {len(class_means)} matrices, each {class_means[0].shape}  ({t_means:.2f}s)")
    # density matrices: trace ≈ 1 (since each pure state has trace 1, average preserves it)
    for c in range(4):
        tr = float(np.real(np.trace(class_means[c])))
        herm = float(np.linalg.norm(class_means[c] - class_means[c].conj().T))
        print(f"  class {c}: trace={tr:.4f}  hermitian_err={herm:.2e}")

    t0 = time.perf_counter()
    train_scores = compute_scores(reducer, tl, class_means, n_qubits=4, num_classes=4)
    t_scores = time.perf_counter() - t0
    print(f"train_scores: {len(train_scores)} entries  ({t_scores:.2f}s)")

    # Reuse train means for val/test
    val_scores = compute_scores(reducer, vl, class_means, n_qubits=4, num_classes=4)
    test_scores = compute_scores(reducer, te, class_means, n_qubits=4, num_classes=4)
    print(f"val_scores={len(val_scores)}  test_scores={len(test_scores)}")

    # WE mask on train, then apply same threshold to val/test
    train_we, thr = make_we_mask(train_scores, percentile=20.0)
    val_we = apply_threshold(val_scores, thr)
    test_we = apply_threshold(test_scores, thr)
    print(f"WE counts (threshold={thr:.4f}): train={len(train_we)}/{len(train_scores)}  "
          f"val={len(val_we)}/{len(val_scores)}  test={len(test_we)}/{len(test_scores)}")

    # Distribution glance
    arr = np.array(list(train_scores.values()))
    print(f"train score: min={arr.min():.4f}  max={arr.max():.4f}  "
          f"mean={arr.mean():.4f}  std={arr.std():.4f}")

    # Per-class score summary
    labels = {idx: int(tds[idx][1]) for idx in train_scores}
    for c in range(4):
        cs = np.array([s for idx, s in train_scores.items() if labels[idx] == c])
        print(f"  class {c}: mean={cs.mean():.4f}  std={cs.std():.4f}  n={len(cs)}")
