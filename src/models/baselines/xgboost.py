"""XGBoost baseline."""

import numpy as np
from xgboost import XGBClassifier


def build_xgb(config: dict = None, scale_pos_weight: float = None) -> XGBClassifier:
    """Build XGBClassifier.

    Args:
        config:           Hyperparameter dict from experiments.yaml.
        scale_pos_weight: n_neg / n_pos ratio for imbalanced datasets.
    """
    cfg = config or {}
    return XGBClassifier(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.05),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        eval_metric=cfg.get("eval_metric", "aucpr"),
        early_stopping_rounds=cfg.get("early_stopping_rounds", 30),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
