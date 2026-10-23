"""Random Forest baseline."""

from sklearn.ensemble import RandomForestClassifier


def build_rf(config: dict = None) -> RandomForestClassifier:
    cfg = config or {}
    return RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", None),
        min_samples_leaf=cfg.get("min_samples_leaf", 2),
        class_weight=cfg.get("class_weight", "balanced"),
        random_state=42,
        n_jobs=-1,
    )
