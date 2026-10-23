"""Loader for the PaySim synthetic mobile-money dataset.

Columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
         nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
Drop:    nameOrig, nameDest (high-cardinality IDs), isFlaggedFraud (rule-based flag)
Label:   isFraud
Note:    Only TRANSFER and CASH_OUT transaction types contain fraud.
"""

import pandas as pd
import numpy as np
from pathlib import Path


FEATURE_TYPES = {
    "numerical": [
        "step", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
    ],
    "categorical": ["type"],
    "temporal":    ["step"],
    "drop":        ["nameOrig", "nameDest", "isFlaggedFraud"],
}


def load_paysim(raw_path: str, config: dict = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load PaySim dataset.

    Args:
        raw_path: Path to PaySim CSV file.
        config:   Optional dataset config dict from datasets.yaml.

    Returns:
        X:             Feature DataFrame.
        y:             Binary label Series (1 = fraud).
        feature_types: Dict with 'numerical', 'categorical', 'temporal' key lists.
    """
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"PaySim dataset not found at: {path}")

    label_col = config["label_col"] if config else "isFraud"
    drop_cols = (config.get("drop_cols") or FEATURE_TYPES["drop"]) if config else FEATURE_TYPES["drop"]

    df = pd.read_csv(path)

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col] + [c for c in drop_cols if c in df.columns])

    return X, y, FEATURE_TYPES
