"""Evaluation metrics for imbalanced fraud detection."""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, brier_score_loss,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    threshold: float = 0.5,
) -> dict:
    """Compute full metric suite for fraud detection.

    Args:
        y_true:    Ground-truth binary labels.
        y_pred:    Predicted binary labels (hard predictions).
        y_prob:    Predicted probabilities for the positive class.
        threshold: Classification threshold (if y_pred not provided).

    Returns:
        Dict of metric name → value.
    """
    if y_pred is None and y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
        "fpr":       fp / max(fp + tn, 1),
        "fnr":       fn / max(fn + tp, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    if y_prob is not None:
        metrics["roc_auc"]   = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"]    = average_precision_score(y_true, y_prob)
        metrics["brier"]     = brier_score_loss(y_true, y_prob)

    return metrics


def format_metrics(metrics: dict) -> str:
    """Return a concise one-line metric summary string."""
    parts = []
    for k in ["f1", "pr_auc", "roc_auc", "precision", "recall", "mcc"]:
        if k in metrics:
            parts.append(f"{k.upper()}={metrics[k]:.4f}")
    return "  ".join(parts)
