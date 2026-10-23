"""Loader for the ULB Credit Card Fraud dataset.

Columns: Time, V1-V28 (PCA), Amount, Class (label)
All columns are numeric. No categorical features.
"""

import pandas as pd
import numpy as np
from pathlib import Path


FEATURE_TYPES = {
    "numerical": ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
    "categorical": [],
    "temporal": ["Time"],
    "drop": [],
}


def load_ulb(raw_path: str, config: dict = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load ULB Credit Card dataset.

    Args:
        raw_path: Path to creditcard.csv.
        config:   Optional dataset config dict from datasets.yaml.

    Returns:
        X:             Feature DataFrame (n_rows × 30).
        y:             Binary label Series (1 = fraud).
        feature_types: Dict with 'numerical', 'categorical', 'temporal' key lists.
    """
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"ULB dataset not found at: {path}")

    df = pd.read_csv(path)

    label_col = config["label_col"] if config else "Class"
    drop_cols = (config.get("drop_cols") or []) if config else []

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col] + [c for c in drop_cols if c in df.columns])

    return X, y, FEATURE_TYPES
