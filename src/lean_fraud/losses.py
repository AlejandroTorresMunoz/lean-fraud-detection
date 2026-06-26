"""Losses for extreme class imbalance (~0.5% fraud).

Focal loss down-weights the easy, abundant negatives so the gradient focuses on the rare fraud
cases — the standard remedy when plain BCE is swamped by the majority class.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss (Lin et al., 2017) on raw logits, mean-reduced.

    alpha weights the positive (fraud) class; gamma is the focusing strength (gamma=0 -> weighted
    BCE). Numerically stable via binary_cross_entropy_with_logits.
    """
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)  # prob of the true class
    loss = ce * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def build_loss(cfg_training: dict) -> "callable":
    """Return a loss(logits, targets) -> scalar from the training config.

    `loss: focal` -> focal_loss_with_logits with the configured alpha/gamma.
    `loss: bce`   -> BCEWithLogits, optionally with a pos_weight set later by the caller.
    """
    kind = cfg_training.get("loss", "focal")
    if kind == "focal":
        alpha = float(cfg_training.get("focal_alpha", 0.25))
        gamma = float(cfg_training.get("focal_gamma", 2.0))
        return lambda logits, targets: focal_loss_with_logits(logits, targets, alpha, gamma)
    if kind == "bce":
        return lambda logits, targets: F.binary_cross_entropy_with_logits(logits, targets)
    raise ValueError(f"Unknown loss {kind!r} (expected 'focal' or 'bce').")
