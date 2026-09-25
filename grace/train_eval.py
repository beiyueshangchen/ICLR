"""Training and evaluation loops for GRACE.

Each learner trajectory is one training unit: the state ``theta`` is unrolled
over the whole sequence, the loss is differentiated with respect to the state at
every step (``create_graph=True``) and the resulting gradient is handed to the
graph-conditioned meta-optimizer.  Sequences inside a batch are averaged, so
``batch_size=1`` reproduces the paper's setup while larger batches remain valid.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

Metrics = Tuple[float, float, float]  # (loss, auc, acc)


def _sanitize_probs_and_labels(all_probs: List[float], all_labels: List[int]):
    """Drop non-finite scores and clip probabilities into ``[0, 1]``."""
    sanitized_probs, sanitized_labels = [], []
    for prob, label in zip(all_probs, all_labels):
        if not math.isfinite(float(prob)) or not math.isfinite(float(label)):
            continue
        sanitized_probs.append(min(max(float(prob), 0.0), 1.0))
        sanitized_labels.append(int(label))
    return sanitized_probs, sanitized_labels


def _compute_metrics(all_probs: List[float], all_labels: List[int], mean_loss: float) -> Metrics:
    all_probs, all_labels = _sanitize_probs_and_labels(all_probs, all_labels)
    if len(all_labels) == 0:
        print("[warn] metrics skipped: no valid samples after sanitization")
        return mean_loss, float("nan"), float("nan")
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_probs])
    return mean_loss, auc, acc


def _masked_state_and_logit(model, theta, item_content_emb, concept_emb, idx, device):
    """Apply the Q-mask, the global concept context and the prediction head."""
    item_vec = model.item_vector(item_content_emb, idx, device)
    q_row = model.Q[idx : idx + 1]
    theta_masked = theta * q_row
    global_concept = model.get_global_concept(theta_masked, concept_emb)
    logit = model.head(theta_masked, item_vec, global_concept)
    return logit, item_vec


def train_one_epoch(
    model,
    loader,
    device,
    optimizer,
    alpha: float = 1.0,
    lambda_1: float = 0.6,
    lambda_2: float = 0.4,
    grad_clip: float = 0.0,
) -> Metrics:
    """Run one epoch of unrolled state updates; return ``(loss, auc, acc)``."""
    model.train()
    all_probs, all_labels = [], []
    epoch_loss = 0.0
    num_steps = 0

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        concept_emb, item_content_emb = model.forward_concept_and_items()
        batch_loss = torch.zeros((), device=device)

        for student_seq in batch:
            theta = torch.zeros(1, model.student_dim, device=device, requires_grad=True)

            sequence_loss = torch.zeros((), device=device)
            reg_loss = torch.zeros((), device=device)
            correct_count = 0

            for i, step in enumerate(student_seq):
                idx = step["problem_idx"]
                y = torch.tensor([step["label"]], dtype=torch.float32, device=device)
                dt = torch.tensor([step["dt"]], dtype=torch.float32, device=device)

                logit, item_vec = _masked_state_and_logit(
                    model, theta, item_content_emb, concept_emb, idx, device
                )
                step_loss = F.binary_cross_entropy_with_logits(logit, y)

                # Gradient of the prediction error with respect to the state; the
                # graph is kept so the meta-optimizer is trained end to end.
                grad_theta = torch.autograd.grad(
                    step_loss, theta, create_graph=True, retain_graph=True
                )[0]

                running_accuracy = correct_count / max(i, 1)
                r_t = torch.tensor([[running_accuracy]], dtype=torch.float32, device=device)
                graph_prop = model.graph_prop if model.use_graph_prop else None

                theta_next, aux = model.meta_opt(
                    theta, grad_theta, item_vec, y, dt, r_t, graph_prop
                )

                reg_loss = reg_loss + alpha * lambda_1 * torch.norm(theta_next - theta, p=2) ** 2
                reg_loss = reg_loss + alpha * lambda_2 * torch.norm(aux["U"], p="fro") ** 2

                sequence_loss = sequence_loss + step_loss
                theta = theta_next

                correct_count += step["label"]
                all_probs.append(torch.sigmoid(logit).detach().cpu().item())
                all_labels.append(step["label"])

            batch_loss = batch_loss + sequence_loss + reg_loss
            epoch_loss += float(sequence_loss.detach()) + float(reg_loss.detach())
            num_steps += len(student_seq)

        batch_loss = batch_loss / len(batch)
        batch_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

    mean_loss = epoch_loss / max(num_steps, 1)
    return _compute_metrics(all_probs, all_labels, mean_loss)


@torch.no_grad()
def evaluate(model, loader, device) -> Metrics:
    """Evaluate next-response prediction; return ``(loss, auc, acc)``."""
    model.eval()
    all_probs, all_labels = [], []
    epoch_loss = 0.0
    num_steps = 0

    concept_emb, item_content_emb = model.forward_concept_and_items()

    for batch in tqdm(loader, desc="Evaluating"):
        for student_seq in batch:
            theta = torch.zeros(1, model.student_dim, device=device)
            correct_count = 0

            for i, step in enumerate(student_seq):
                idx = step["problem_idx"]
                y = step["label"]
                y_t = torch.tensor([y], dtype=torch.float32, device=device)
                dt = torch.tensor([step["dt"]], dtype=torch.float32, device=device)

                logit, item_vec = _masked_state_and_logit(
                    model, theta, item_content_emb, concept_emb, idx, device
                )
                epoch_loss += float(F.binary_cross_entropy_with_logits(logit, y_t).detach())
                num_steps += 1

                all_probs.append(torch.sigmoid(logit).item())
                all_labels.append(y)

                # Recompute the state gradient for the transition, this time
                # without building a graph for it.
                with torch.enable_grad():
                    theta_grad = theta.clone().requires_grad_(True)
                    q_row = model.Q[idx : idx + 1]
                    theta_masked = theta_grad * q_row
                    global_concept = model.get_global_concept(theta_masked, concept_emb)
                    inner_loss = F.binary_cross_entropy_with_logits(
                        model.head(theta_masked, item_vec, global_concept), y_t
                    )
                    grad_theta = torch.autograd.grad(inner_loss, theta_grad)[0]

                running_accuracy = correct_count / max(i, 1)
                r_t = torch.tensor([[running_accuracy]], dtype=torch.float32, device=device)
                graph_prop = model.graph_prop if model.use_graph_prop else None
                theta, _ = model.meta_opt(
                    theta_grad.detach(),
                    grad_theta.detach(),
                    item_vec.detach(),
                    y_t,
                    dt,
                    r_t,
                    graph_prop,
                )
                theta = theta.detach()
                correct_count += y

    mean_loss = epoch_loss / max(num_steps, 1)
    return _compute_metrics(all_probs, all_labels, mean_loss)
