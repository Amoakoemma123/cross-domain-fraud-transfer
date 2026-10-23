"""Cross-domain evaluation utilities."""

import numpy as np
import pandas as pd
from typing import Callable
from .metrics import compute_metrics


def evaluate_cross_domain(
    model,
    X_target: np.ndarray,
    y_target: np.ndarray,
    predict_proba_fn: Callable = None,
    threshold: float = 0.5,
) -> dict:
    """Evaluate a source-trained model on a target domain.

    Args:
        model:           Fitted sklearn-compatible estimator or callable.
        X_target:        Target domain features (aligned with source feature space).
        y_target:        Target domain labels.
        predict_proba_fn: Optional override for probability prediction.
        threshold:       Decision threshold.

    Returns:
        Metrics dict.
    """
    if predict_proba_fn is not None:
        y_prob = predict_proba_fn(X_target)
    elif hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_target)[:, 1]
    else:
        y_prob = None

    if y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_target)

    return compute_metrics(y_target, y_pred, y_prob, threshold)


def build_cross_domain_matrix(results: dict) -> pd.DataFrame:
    """Convert nested results dict to a source × target metric DataFrame.

    Args:
        results: {source_dataset: {target_dataset: metrics_dict}}

    Returns:
        DataFrame with sources as rows and targets as columns (F1 values).
    """
    sources = sorted(results.keys())
    targets = sorted({t for src in results.values() for t in src.keys()})
    matrix = pd.DataFrame(index=sources, columns=targets, dtype=float)

    for src, tgt_dict in results.items():
        for tgt, m in tgt_dict.items():
            matrix.loc[src, tgt] = round(m.get("f1", float("nan")), 4)

    return matrix
