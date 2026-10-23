"""LightGBM baseline."""

from lightgbm import LGBMClassifier


def build_lgbm(config: dict = None) -> LGBMClassifier:
    cfg = config or {}
    return LGBMClassifier(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=cfg.get("max_depth", -1),
        learning_rate=cfg.get("learning_rate", 0.05),
        num_leaves=cfg.get("num_leaves", 63),
        class_weight=cfg.get("class_weight", "balanced"),
        early_stopping_round=cfg.get("early_stopping_round", 30),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
