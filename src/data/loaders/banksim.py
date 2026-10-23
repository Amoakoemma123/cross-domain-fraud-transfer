"""Loader for the BankSim synthetic banking dataset.

File layout in raw_path directory:
  bs140513_032310.csv     — primary transaction records (use for all ML tasks)
  bsNET140513_032310.csv  — same transactions in customer-merchant graph format

Primary file columns:
  step, customer, age, gender, zipcodeOri, merchant, zipMerchant, category, amount, fraud

Network file columns:
  Source (customer_id), Target (merchant_id), Weight (amount), typeTrans (category), fraud
  → 4,112 unique customers, 50 unique merchants, same 594,643 rows / 7,200 fraud cases

Known quirks:
  - All string columns are single-quoted (e.g. '4', 'es_food', 'M') — stripped on load
  - Network file Source/Target/typeTrans also single-quoted — stripped on load
  - BankSim is the external validation dataset (not used as primary training source)
"""

import pandas as pd
import numpy as np
from pathlib import Path


FEATURE_TYPES = {
    "numerical":   ["step", "amount"],
    "categorical": ["age", "gender", "category"],
    "temporal":    ["step"],
    "drop":        ["customer", "merchant", "zipcodeOri", "zipMerchant"],
}

NETWORK_FEATURE_TYPES = {
    "numerical":   ["Weight"],
    "categorical": ["typeTrans"],
    "node_cols":   ["Source", "Target"],
    "drop":        [],
}


def _strip_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing single quotes from all object columns."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip("'")
    return df


def load_banksim(
    raw_path: str,
    config: dict = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load BankSim primary transaction file.

    Args:
        raw_path: Path to bs140513_032310.csv.
        config:   Optional dataset config dict from datasets.yaml.

    Returns:
        X:             Feature DataFrame (594,643 × 5 after dropping IDs).
        y:             Binary label Series (1 = fraud).
        feature_types: Dict with 'numerical', 'categorical', 'temporal' key lists.
    """
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"BankSim transaction file not found: {path}")

    label_col = config["label_col"] if config else "fraud"
    drop_cols = (config.get("drop_cols") or FEATURE_TYPES["drop"]) if config else FEATURE_TYPES["drop"]

    df = pd.read_csv(path)
    df = _strip_quotes(df)

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col] + [c for c in drop_cols if c in df.columns])

    return X, y, FEATURE_TYPES


def load_banksim_network(raw_dir: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load BankSim network (graph) file.

    Returns the customer-merchant bipartite graph representation of the same
    594,643 transactions. Useful for deriving graph-based features such as:
      - customer transaction degree / fraud rate
      - merchant fraud rate
      - weighted customer-merchant edge features

    Args:
        raw_dir: Directory containing bsNET140513_032310.csv.
                 Can also be a direct path to the file.

    Returns:
        X:             Network feature DataFrame (Weight + encoded typeTrans).
        y:             Binary label Series (1 = fraud).
        feature_types: Dict with 'numerical', 'categorical', 'node_cols' key lists.
    """
    p = Path(raw_dir)
    net_path = (p / "bsNET140513_032310.csv") if p.is_dir() else p
    if not net_path.exists():
        raise FileNotFoundError(f"BankSim network file not found: {net_path}")

    df = pd.read_csv(net_path)
    df = _strip_quotes(df)

    label_col = "fraud"
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    return X, y, NETWORK_FEATURE_TYPES
