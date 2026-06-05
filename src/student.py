"""Quantum student: TutorialReducer (Stage A) + AmplitudeEmbedding (Stage B)
+ StronglyEntanglingLayers (Stage C) + classical head.

Batching strategy
-----------------
We wrap the QNode in `qml.qnn.TorchLayer`, which feeds batched inputs to the
QNode. With `default.qubit` + `diff_method="backprop"`, batches are vectorised
at the simulator level (real broadcasting, fast). With `lightning.qubit` +
`diff_method="adjoint"`, TorchLayer loops over the batch dimension internally
(slower but exact gradients).

Use `device_kind="default"` for toy speed; `"lightning"` is kept as a fallback.
"""
from __future__ import annotations

from typing import Literal

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class TutorialReducer(nn.Module):
    """Stage A (main, amplitude encoding): adaptive avg-pool 28x28 -> sqrt(2^n) x sqrt(2^n).

    No trainable parameters. 4-qubit -> 4x4=16; 8-qubit -> 16x16=256.
    """

    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.target_dim = 2 ** n_qubits
        side = int(round(self.target_dim ** 0.5))
        if side * side != self.target_dim:
            raise ValueError(
                f"n_qubits={n_qubits} -> target_dim={self.target_dim} is not a perfect square; "
                "TutorialReducer only supports even-qubit counts."
            )
        self.target_side = side

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.adaptive_avg_pool2d(x, (self.target_side, self.target_side))
        return x.flatten(1)


class AngleReducer(nn.Module):
    """Stage A (angle encoding): adaptive avg-pool 28x28 -> n_qubits floats, scaled to [0, π].

    Each output dim becomes one R_Y rotation angle. No trainable parameters.
    4-qubit -> 4 angles; 8-qubit -> 8 angles. Massively smaller representation
    than amplitude (which gets 2^n floats).
    """

    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.adaptive_avg_pool2d(x, (1, self.n_qubits))  # (B, C, 1, n_qubits)
        x = x.flatten(1)  # (B, n_qubits)
        return x * 3.14159265


def _make_qnode(n_qubits: int, n_meas: int, device_kind: str, encoding: str = "amplitude"):
    if device_kind == "default":
        dev = qml.device("default.qubit", wires=n_qubits)
        diff = "backprop"
    elif device_kind == "lightning":
        dev = qml.device("lightning.qubit", wires=n_qubits)
        diff = "adjoint"
    else:
        raise ValueError(f"Unknown device_kind: {device_kind}")

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def circuit(inputs, weights):
        # Stage B: quantum encoding
        if encoding == "amplitude":
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        elif encoding == "angle":
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        # Stage C: PQC
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_meas)]

    return circuit


class QuantumStudent(nn.Module):
    """Stage A + B + C end-to-end module."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        num_classes: int = 4,
        reducer_type: Literal["tutorial", "angle", "pca"] = "tutorial",
        device_kind: Literal["default", "lightning"] = "default",
        pca_reducer: nn.Module | None = None,
        encoding: Literal["amplitude", "angle"] = "amplitude",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_meas = min(n_qubits, num_classes)
        self.encoding = encoding

        if reducer_type == "tutorial":
            self.reducer = TutorialReducer(n_qubits)
        elif reducer_type == "angle":
            self.reducer = AngleReducer(n_qubits)
        elif reducer_type == "pca":
            if pca_reducer is None:
                raise ValueError("reducer_type='pca' requires a fitted pca_reducer instance.")
            self.reducer = pca_reducer
        else:
            raise ValueError(f"Unknown reducer_type: {reducer_type}")

        circuit = _make_qnode(n_qubits, self.n_meas, device_kind, encoding=encoding)
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": weight_shape})
        self.head = nn.Linear(self.n_meas, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reducer(x)              # (B, 2^n)
        q = self.qlayer(z)               # (B, n_meas)
        return self.head(q)


if __name__ == "__main__":
    import time
    from data import load_data

    torch.manual_seed(42)
    tl, _, _, _, _, _ = load_data("mnist", batch_size=8, seed=42)
    xb, yb, ib = next(iter(tl))
    print(f"input batch: x={tuple(xb.shape)}")

    # 1) TutorialReducer shape check
    red = TutorialReducer(n_qubits=4)
    z = red(xb)
    print(f"reducer 4q: x{tuple(xb.shape)} -> z{tuple(z.shape)}  (expect (8,16))")
    assert z.shape == (8, 16)

    red8 = TutorialReducer(n_qubits=8)
    z8 = red8(xb)
    print(f"reducer 8q: x{tuple(xb.shape)} -> z{tuple(z8.shape)}  (expect (8,256))")
    assert z8.shape == (8, 256)

    # 2) QuantumStudent forward — default.qubit (broadcast)
    print("\n-- default.qubit (backprop, broadcast) --")
    model_d = QuantumStudent(n_qubits=4, n_layers=2, num_classes=4, device_kind="default")
    t0 = time.perf_counter()
    out_d = model_d(xb)
    t_d = time.perf_counter() - t0
    print(f"forward batch=8: out={tuple(out_d.shape)}  time={t_d * 1000:.1f}ms")
    assert out_d.shape == (8, 4)

    # 2b) Backward sanity
    loss = F.cross_entropy(out_d, yb)
    loss.backward()
    head_grad = model_d.head.weight.grad
    qw_grad = next(p.grad for n, p in model_d.qlayer.named_parameters() if "weights" in n)
    print(f"backward OK: head_grad_norm={head_grad.norm():.4f}  qweights_grad_norm={qw_grad.norm():.4f}")
    assert head_grad is not None and qw_grad is not None

    # 3) QuantumStudent forward — lightning.qubit (loop) — speed comparison
    print("\n-- lightning.qubit (adjoint, internal loop) --")
    torch.manual_seed(42)
    model_l = QuantumStudent(n_qubits=4, n_layers=2, num_classes=4, device_kind="lightning")
    t0 = time.perf_counter()
    out_l = model_l(xb)
    t_l = time.perf_counter() - t0
    print(f"forward batch=8: out={tuple(out_l.shape)}  time={t_l * 1000:.1f}ms")
    assert out_l.shape == (8, 4)

    print(f"\nspeed ratio (default / lightning): {t_d / t_l:.2f}x")
    print(f"  default = {t_d*1000:.1f}ms  lightning = {t_l*1000:.1f}ms  for batch=8, 4q, 2 layers")
