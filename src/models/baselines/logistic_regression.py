"""Logistic Regression baseline."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_lr(config: dict = None) -> LogisticRegression:
    cfg = config or {}
    return LogisticRegression(
        C=cfg.get("C", 1.0),
        max_iter=cfg.get("max_iter", 1000),
        solver=cfg.get("solver", "lbfgs"),
        class_weight=cfg.get("class_weight", "balanced"),
        random_state=42,
        n_jobs=-1,
    )
