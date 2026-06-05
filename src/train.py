"""Knowledge-distillation training loop for Q-RAD.

Method IDs used by the public pipeline:

  Method | L_task | L_KD             | Trained samples | Paper label
  -------|--------|------------------|-----------------|---------------------
  1      | all    | -                | all             | Baseline / Fair
  3      | all    | uniform on all   | all             | Baseline_KD
  6      | all    | KD on PE only    | all             | PQC_PE
  8      | all    | KD on CE only    | all             | PQC_CE

`is_we` / `we_mask` are kept as internal variable names; they correspond to
the CE region (Compression-affected Encoding) as defined by the paper's
20th-percentile HEM cutoff.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_sample_kd(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    T: float) -> torch.Tensor:
    """KL(p_T^T || p_S^T) * T^2 per sample, summed over classes."""
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1) * (T * T)


def compute_loss(
    method_id: int,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    y: torch.Tensor,
    is_we: torch.Tensor,        # bool, shape (B,) — True for CE-region samples
    lambda_kd: float = 0.1,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, task_component, kd_component).

    task_component + kd_component == total_loss; logging the two parts
    separately reveals when the KD term overwhelms the task term.
    """
    task_per = F.cross_entropy(student_logits, y, reduction="none")
    kd_per = _per_sample_kd(student_logits, teacher_logits, temperature)
    is_non_we = ~is_we
    n_non_we = int(is_non_we.sum().item())
    n_we = int(is_we.sum().item())
    device = student_logits.device
    zero = torch.zeros((), device=device)

    if method_id == 1:
        # Baseline / Fair: task loss only, no KD
        task_term = task_per.mean()
        kd_term = zero
    elif method_id == 3:
        # Baseline_KD: uniform KD on all samples
        task_term = task_per.mean()
        kd_term = lambda_kd * kd_per.mean()
    elif method_id == 6:
        # PQC_PE: task on all + KD restricted to PE samples (non-CE)
        task_term = task_per.mean()
        kd_term = lambda_kd * kd_per[is_non_we].mean() if n_non_we > 0 else zero
    elif method_id == 8:
        # PQC_CE: task on all + KD restricted to CE samples
        task_term = task_per.mean()
        kd_term = lambda_kd * kd_per[is_we].mean() if n_we > 0 else zero
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
    """Overall test accuracy + CE-region (acc_we) / PE-region (acc_non_we) accuracy.

    Key names `acc_we` / `acc_non_we` are kept for backward-compatibility with
    the per-seed JSON files; in the paper they correspond to CE and PE regions.
    """
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
                    n_we += 1; c_we += int(ok)
                else:
                    n_nw += 1; c_nw += int(ok)
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
    n_epochs: int = 50,
    lr: float = 1e-3,
    lambda_kd: float = 0.1,
    temperature: float = 2.0,
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
            loss, task_c, kd_c = compute_loss(
                method_id, student_logits, teacher_logits, y, is_we,
                lambda_kd=lambda_kd, temperature=temperature,
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
                  f"val_all={val_metrics['acc_all']:.3f}  val_CE={we_str}  val_PE={nw_str}")

    return history
