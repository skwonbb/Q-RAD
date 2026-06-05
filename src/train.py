"""Unified training loop for the ablation methods (toy guide §5 + reverse variants).

Method ID | name                             | L_task    | L_KD                  | Data
----------|----------------------------------|-----------|-----------------------|------
1         | QNN-only (no KD)                 | all       | -                     | all
2         | QNN-only + WE removed (sanity)   | non-WE    | -                     | non-WE
3         | Uniform KD (Hasan baseline)      | all       | uniform all           | all
4         | Q-SED-Base                       | non-WE    | non-WE                | non-WE
5         | Confidence-weighted KD           | all       | weighted all          | all
6         | Q-SED-Plus                       | all       | non-WE only           | all
7         | Q-SED-Base-rev (Stage 2-3)       | WE only   | WE only               | WE only
8         | Q-SED-Plus-rev (Stage 2-3)       | all       | WE only               | all
9         | Q-SED-Adaptive (Stage 2-7)       | all       | per-sample lambda     | all
10        | Q-SED-Continuous (Stage 2-8)     | all       | lambda(x) = f(score)  | all
11        | Q-SED-Convex (Stage 2-10)        | (1-a(x))  | a(x), convex          | all
13        | Q-SED-DualSweep (Stage 2-12)     | (1-a(x))  | a(x) + per-sample T   | all
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_sample_kd(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    T) -> torch.Tensor:
    """KL(p_T^T || p_S^T) * T^2 per sample, summed over classes.

    T may be a Python scalar (uniform) or a 1-D tensor of shape (B,) for
    sample-wise temperature (Method 13).
    """
    if torch.is_tensor(T) and T.dim() == 1:
        T_div = T.unsqueeze(-1)        # (B, 1) for broadcasting over class dim
        T_sq = T * T                    # (B,)
    else:
        T_div = T
        T_sq = T * T
    log_p_s = F.log_softmax(student_logits / T_div, dim=-1)
    p_t = F.softmax(teacher_logits / T_div, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1) * T_sq


def compute_loss(
    method_id: int,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    y: torch.Tensor,
    is_we: torch.Tensor,        # bool, shape (B,)
    lambda_kd: float = 0.5,
    temperature: float = 4.0,
    lambda_kd_low: float | None = None,    # Method 9 only
    lambda_kd_high: float | None = None,   # Method 9 only
    lambda_per_sample: torch.Tensor | None = None,  # Method 10 only, shape (B,)
    alpha_low: float | None = None,         # Method 11 binary
    alpha_high: float | None = None,        # Method 11 binary
    alpha_per_sample: torch.Tensor | None = None,  # Method 11 continuous, shape (B,)
    temperature_per_sample: torch.Tensor | None = None,  # Method 13, shape (B,)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, task_component, kd_component).

    `task_component + kd_component == total_loss` (when both are nonzero), so
    logging the two parts separately reveals when the KD term overwhelms the
    task term — the failure mode we hit at lambda=0.5, T=4.
    """
    task_per = F.cross_entropy(student_logits, y, reduction="none")
    kd_per = _per_sample_kd(student_logits, teacher_logits, temperature)
    is_non_we = ~is_we
    n_non_we = int(is_non_we.sum().item())
    device = student_logits.device
    zero = torch.zeros((), device=device)

    if method_id == 1:
        task_term = task_per.mean()
        kd_term = zero
    elif method_id == 2:
        if n_non_we == 0:
            task_term = torch.zeros((), device=device, requires_grad=True)
        else:
            task_term = task_per[is_non_we].mean()
        kd_term = zero
    elif method_id == 3:
        task_term = task_per.mean()
        kd_term = lambda_kd * kd_per.mean()
    elif method_id == 4:
        if n_non_we == 0:
            task_term = torch.zeros((), device=device, requires_grad=True)
            kd_term = zero
        else:
            task_term = task_per[is_non_we].mean()
            kd_term = lambda_kd * kd_per[is_non_we].mean()
    elif method_id == 5:
        with torch.no_grad():
            confidence = F.softmax(teacher_logits, dim=-1).max(dim=-1).values
        task_term = task_per.mean()
        kd_term = lambda_kd * (confidence * kd_per).mean()
    elif method_id == 6:
        task_term = task_per.mean()
        kd_term = lambda_kd * kd_per[is_non_we].mean() if n_non_we > 0 else zero
    elif method_id == 7:
        # Reverse of method 4: WE-only task + KD, non-WE entirely removed.
        n_we = int(is_we.sum().item())
        if n_we == 0:
            task_term = torch.zeros((), device=device, requires_grad=True)
            kd_term = zero
        else:
            task_term = task_per[is_we].mean()
            kd_term = lambda_kd * kd_per[is_we].mean()
    elif method_id == 8:
        # Reverse of method 6: task on all, KD only on WE samples.
        n_we = int(is_we.sum().item())
        task_term = task_per.mean()
        kd_term = lambda_kd * kd_per[is_we].mean() if n_we > 0 else zero
    elif method_id == 9:
        # Q-SED-Adaptive: per-sample lambda. lambda_high on WE, lambda_low on non-WE.
        # task on all (kept full), KD weighted per sample by lambda(x).
        if lambda_kd_low is None or lambda_kd_high is None:
            raise ValueError("Method 9 requires lambda_kd_low and lambda_kd_high.")
        lam_per = torch.where(
            is_we,
            torch.full_like(task_per, float(lambda_kd_high)),
            torch.full_like(task_per, float(lambda_kd_low)),
        )
        task_term = task_per.mean()
        kd_term = (lam_per * kd_per).mean()
    elif method_id == 10:
        # Q-SED-Continuous: lambda(x) = f(score(x)) for all samples.
        # The caller computes lambda_per_sample externally (since the mapping
        # depends on the precomputed score for each sample's index).
        if lambda_per_sample is None:
            raise ValueError("Method 10 requires lambda_per_sample tensor (shape (B,)).")
        if lambda_per_sample.shape != task_per.shape:
            raise ValueError(f"lambda_per_sample shape {lambda_per_sample.shape} "
                             f"!= task_per shape {task_per.shape}")
        task_term = task_per.mean()
        kd_term = (lambda_per_sample.to(device) * kd_per).mean()
    elif method_id == 11:
        # Q-SED-Convex: L(x) = (1 - a(x)) * L_task(x) + a(x) * KD(x).
        # Two flavours: binary (alpha_low/high) and continuous (alpha_per_sample).
        if alpha_per_sample is not None:
            if alpha_per_sample.shape != task_per.shape:
                raise ValueError(f"alpha_per_sample shape {alpha_per_sample.shape} "
                                 f"!= task_per shape {task_per.shape}")
            a = alpha_per_sample.to(device)
        elif alpha_low is not None and alpha_high is not None:
            a = torch.where(
                is_we,
                torch.full_like(task_per, float(alpha_high)),
                torch.full_like(task_per, float(alpha_low)),
            )
        else:
            raise ValueError(
                "Method 11 requires either (alpha_low, alpha_high) or alpha_per_sample.")
        task_term = ((1.0 - a) * task_per).mean()
        kd_term = (a * kd_per).mean()
    elif method_id == 13:
        # Q-SED-DualSweep: per-sample alpha AND per-sample temperature.
        # KD term must be recomputed with per-sample T (since the global kd_per
        # used at the top of this fn assumed scalar `temperature`).
        if alpha_per_sample is None:
            raise ValueError("Method 13 requires alpha_per_sample (shape (B,)).")
        if temperature_per_sample is None:
            raise ValueError("Method 13 requires temperature_per_sample (shape (B,)).")
        if alpha_per_sample.shape != task_per.shape:
            raise ValueError("alpha_per_sample shape mismatch.")
        if temperature_per_sample.shape != task_per.shape:
            raise ValueError("temperature_per_sample shape mismatch.")
        a = alpha_per_sample.to(device)
        T_vec = temperature_per_sample.to(device)
        kd_per_T = _per_sample_kd(student_logits, teacher_logits, T_vec)
        task_term = ((1.0 - a) * task_per).mean()
        kd_term = (a * kd_per_T).mean()
    else:
        raise ValueError(f"Unknown method_id: {method_id}")

    total = task_term + kd_term
    return total, task_term.detach(), kd_term.detach()


def _is_we_tensor(idx: torch.Tensor, we_mask: set[int], device) -> torch.Tensor:
    return torch.tensor([int(i) in we_mask for i in idx.tolist()], dtype=torch.bool, device=device)


def evaluate(
    student: nn.Module,
    loader,
    we_mask: set[int],
    device: str = "cpu",
) -> dict:
    """Overall test accuracy + WE-region / non-WE-region accuracy."""
    student.eval()
    n_all = n_we = n_nw = 0
    c_all = c_we = c_nw = 0
    with torch.no_grad():
        for x, y, idx in loader:
            x, y = x.to(device), y.to(device)
            pred = student(x).argmax(-1)
            correct = (pred == y).cpu()
            for i in range(x.size(0)):
                ok = bool(correct[i].item())
                in_we = int(idx[i].item()) in we_mask
                n_all += 1
                c_all += int(ok)
                if in_we:
                    n_we += 1
                    c_we += int(ok)
                else:
                    n_nw += 1
                    c_nw += int(ok)
    return {
        "acc_all": c_all / max(n_all, 1),
        "acc_we": (c_we / n_we) if n_we > 0 else float("nan"),
        "acc_non_we": (c_nw / n_nw) if n_nw > 0 else float("nan"),
        "n_all": n_all,
        "n_we": n_we,
        "n_non_we": n_nw,
    }


def train_method(
    method_id: int,
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    val_loader,
    train_we_mask: set[int],
    val_we_mask: set[int],
    n_epochs: int = 15,
    lr: float = 1e-3,
    lambda_kd: float = 0.5,
    temperature: float = 4.0,
    lambda_kd_low: float | None = None,
    lambda_kd_high: float | None = None,
    lambda_per_sample_dict: dict[int, float] | None = None,  # Method 10 only
    alpha_low: float | None = None,                # Method 11 binary
    alpha_high: float | None = None,               # Method 11 binary
    alpha_per_sample_dict: dict[int, float] | None = None,  # Method 11 continuous
    temperature_per_sample_dict: dict[int, float] | None = None,  # Method 13
    device: str = "cpu",
    log_every: int = 1,
) -> dict:
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    history = {"train_loss": [], "task_loss": [], "kd_loss": [],
               "val_acc_all": [], "val_acc_we": [], "val_acc_non_we": []}

    for ep in range(n_epochs):
        student.train()
        loss_sum = task_sum = kd_sum = 0.0
        n_batches = 0
        for x, y, idx in train_loader:
            x, y = x.to(device), y.to(device)
            is_we = _is_we_tensor(idx, train_we_mask, device)
            optimizer.zero_grad()
            student_logits = student(x)
            with torch.no_grad():
                teacher_logits = teacher(x)
            lam_per_sample = None
            if method_id == 10:
                if lambda_per_sample_dict is None:
                    raise ValueError("Method 10 requires lambda_per_sample_dict.")
                lam_per_sample = torch.tensor(
                    [lambda_per_sample_dict[int(i)] for i in idx.tolist()],
                    dtype=torch.float32, device=device,
                )
            alpha_per_sample = None
            if method_id in (11, 13) and alpha_per_sample_dict is not None:
                alpha_per_sample = torch.tensor(
                    [alpha_per_sample_dict[int(i)] for i in idx.tolist()],
                    dtype=torch.float32, device=device,
                )
            temperature_per_sample = None
            if method_id == 13:
                if temperature_per_sample_dict is None:
                    raise ValueError("Method 13 requires temperature_per_sample_dict.")
                temperature_per_sample = torch.tensor(
                    [temperature_per_sample_dict[int(i)] for i in idx.tolist()],
                    dtype=torch.float32, device=device,
                )
            loss, task_c, kd_c = compute_loss(
                method_id, student_logits, teacher_logits, y, is_we,
                lambda_kd, temperature,
                lambda_kd_low=lambda_kd_low, lambda_kd_high=lambda_kd_high,
                lambda_per_sample=lam_per_sample,
                alpha_low=alpha_low, alpha_high=alpha_high,
                alpha_per_sample=alpha_per_sample,
                temperature_per_sample=temperature_per_sample,
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            task_sum += float(task_c.item())
            kd_sum += float(kd_c.item())
            n_batches += 1
        train_loss = loss_sum / max(n_batches, 1)
        task_avg = task_sum / max(n_batches, 1)
        kd_avg = kd_sum / max(n_batches, 1)

        val_metrics = evaluate(student, val_loader, val_we_mask, device)
        history["train_loss"].append(train_loss)
        history["task_loss"].append(task_avg)
        history["kd_loss"].append(kd_avg)
        history["val_acc_all"].append(val_metrics["acc_all"])
        history["val_acc_we"].append(val_metrics["acc_we"])
        history["val_acc_non_we"].append(val_metrics["acc_non_we"])
        if (ep + 1) % log_every == 0:
            we_str = f"{val_metrics['acc_we']:.3f}" if val_metrics["n_we"] > 0 else "  n/a"
            nw_str = f"{val_metrics['acc_non_we']:.3f}" if val_metrics["n_non_we"] > 0 else "  n/a"
            ratio = (kd_avg / task_avg) if task_avg > 1e-9 else float("nan")
            print(f"  [m{method_id}] ep {ep + 1:2d}/{n_epochs}  "
                  f"task={task_avg:.3f}  kd={kd_avg:.3f}  ratio={ratio:.2f}  "
                  f"val_all={val_metrics['acc_all']:.3f}  val_we={we_str}  val_nw={nw_str}")

    return history


if __name__ == "__main__":
    # Smoke test: run compute_loss for each method on synthetic logits.
    torch.manual_seed(0)
    B, C = 8, 4
    student_logits = torch.randn(B, C, requires_grad=True)
    teacher_logits = torch.randn(B, C)
    y = torch.randint(0, C, (B,))
    is_we = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)

    for m in (1, 2, 3, 4, 5, 6, 7, 8):
        loss, task_c, kd_c = compute_loss(m, student_logits, teacher_logits, y, is_we)
        student_logits.grad = None
        loss.backward(retain_graph=True)
        print(f"method {m}: total={float(loss.detach()):.4f}  "
              f"task={float(task_c):.4f}  kd={float(kd_c):.4f}  "
              f"grad_norm={student_logits.grad.norm():.4f}")
    # Method 9 with adaptive (lambda_low=0.05, lambda_high=0.5)
    loss, task_c, kd_c = compute_loss(9, student_logits, teacher_logits, y, is_we,
                                       lambda_kd_low=0.05, lambda_kd_high=0.5)
    student_logits.grad = None
    loss.backward(retain_graph=True)
    print(f"method 9 (low=0.05, high=0.5): total={float(loss.detach()):.4f}  "
          f"task={float(task_c):.4f}  kd={float(kd_c):.4f}  "
          f"grad_norm={student_logits.grad.norm():.4f}")
    # Sanity: when low == high == lambda_kd, M9 should match M3 (modulo numerical)
    loss_m3, t3, k3 = compute_loss(3, student_logits, teacher_logits, y, is_we,
                                    lambda_kd=0.3)
    loss_m9, t9, k9 = compute_loss(9, student_logits, teacher_logits, y, is_we,
                                    lambda_kd_low=0.3, lambda_kd_high=0.3)
    print(f"M9(low=high=0.3) == M3(lam=0.3)? "
          f"task: {float(t3):.5f} vs {float(t9):.5f}, "
          f"kd: {float(k3):.5f} vs {float(k9):.5f}")
    # Method 10 with custom lambda_per_sample tensor
    lam_per = torch.tensor([0.5, 0.05, 0.5, 0.05, 0.5, 0.05, 0.5, 0.05])
    loss, task_c, kd_c = compute_loss(10, student_logits, teacher_logits, y, is_we,
                                       lambda_per_sample=lam_per)
    student_logits.grad = None
    loss.backward(retain_graph=True)
    print(f"method 10 (custom lam_per): total={float(loss.detach()):.4f}  "
          f"task={float(task_c):.4f}  kd={float(kd_c):.4f}")
    # Sanity: M10 with constant lam_per == M3
    lam_const = torch.full((8,), 0.3)
    loss_m10, t10, k10 = compute_loss(10, student_logits, teacher_logits, y, is_we,
                                       lambda_per_sample=lam_const)
    print(f"M10(const 0.3) == M3(lam=0.3)? "
          f"task: {float(t3):.5f} vs {float(t10):.5f}, "
          f"kd: {float(k3):.5f} vs {float(k10):.5f}")
    # Method 11 binary
    loss, task_c, kd_c = compute_loss(11, student_logits, teacher_logits, y, is_we,
                                       alpha_low=0.05, alpha_high=0.5)
    student_logits.grad = None
    loss.backward(retain_graph=True)
    print(f"method 11 binary (0.05, 0.5): total={float(loss.detach()):.4f}  "
          f"task={float(task_c):.4f}  kd={float(kd_c):.4f}  "
          f"grad_norm={student_logits.grad.norm():.4f}")
    # Sanity: M11 with alpha_low=alpha_high=0 == M1 (no KD, full task)
    loss_m11_zero, t11z, k11z = compute_loss(11, student_logits, teacher_logits, y, is_we,
                                              alpha_low=0.0, alpha_high=0.0)
    loss_m1, t1, k1 = compute_loss(1, student_logits, teacher_logits, y, is_we)
    print(f"M11(0,0) == M1? task: {float(t1):.5f} vs {float(t11z):.5f},  "
          f"kd: {float(k1):.5f} vs {float(k11z):.5f}")
    # M11 continuous
    alpha_per = torch.tensor([0.5, 0.05, 0.5, 0.05, 0.5, 0.05, 0.5, 0.05])
    loss, task_c, kd_c = compute_loss(11, student_logits, teacher_logits, y, is_we,
                                       alpha_per_sample=alpha_per)
    print(f"method 11 continuous (per-sample): total={float(loss.detach()):.4f}  "
          f"task={float(task_c):.4f}  kd={float(kd_c):.4f}")
    # Method 13 (per-sample alpha + per-sample T)
    T_per = torch.tensor([4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0, 1.0])
    loss, task_c, kd_c = compute_loss(13, student_logits, teacher_logits, y, is_we,
                                       alpha_per_sample=alpha_per,
                                       temperature_per_sample=T_per)
    student_logits.grad = None
    loss.backward(retain_graph=True)
    print(f"method 13 (alpha_per + T_per): total={float(loss.detach()):.4f}  "
          f"task={float(task_c):.4f}  kd={float(kd_c):.4f}  "
          f"grad_norm={student_logits.grad.norm():.4f}")
    # Sanity: M13 with constant T_per equals M11 with same alpha at that T
    T_const = torch.full((8,), 4.0)
    loss_m13, t13, k13 = compute_loss(13, student_logits, teacher_logits, y, is_we,
                                       alpha_per_sample=alpha_per,
                                       temperature_per_sample=T_const)
    loss_m11_T4, t11T4, k11T4 = compute_loss(11, student_logits, teacher_logits, y, is_we,
                                              alpha_per_sample=alpha_per, temperature=4.0)
    print(f"M13(alpha, T=const 4) == M11(alpha, T=4)? "
          f"task: {float(t11T4):.5f} vs {float(t13):.5f},  "
          f"kd: {float(k11T4):.5f} vs {float(k13):.5f}")
