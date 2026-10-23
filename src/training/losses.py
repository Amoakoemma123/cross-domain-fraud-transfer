"""Custom loss functions for imbalanced fraud detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce_loss(pos_weight: float = None):
    """Binary cross-entropy with optional positive-class weighting."""
    if pos_weight is not None:
        weight = torch.tensor([pos_weight])
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    return nn.BCEWithLogitsLoss()


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss for extreme class imbalance.

    Reference: Lin et al. (2017), "Focal Loss for Dense Object Detection".
    """
    def _focal(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** gamma * bce
        return loss.mean()

    return _focal
