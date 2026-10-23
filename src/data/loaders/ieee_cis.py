"""Loader for the IEEE-CIS Fraud Detection dataset.

File layout in raw_path directory:
  train_transaction.csv  — 590,540 rows × 394 cols  (includes isFraud label)
  train_identity.csv     — 144,233 rows × 41 cols   (id_01…id_38, DeviceType, DeviceInfo)
  test_transaction.csv   —  506,691 rows × 393 cols  (no isFraud — unlabeled)
  test_identity.csv      —  141,907 rows × 41 cols   (id-01…id-38 — NOTE: dashes not underscores)
  sample_submission.csv  — Kaggle artifact, ignored

Known quirks:
  - train_identity uses underscores:  id_01, id_02, …
  - test_identity uses dashes:        id-01, id-02, …  → normalised to underscores on load
  - ~414 of 432 joined columns have missing values; 196 cols are >50% missing
  - All identity columns are missing for ~76% of transactions (left-join NaNs)
  - DeviceInfo has 1,786 unique values — treated as high-cardinality categorical
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Categorical columns in transaction file
TXN_CATEGORICAL = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

# Categorical columns in identity file (normalised to underscore naming)
IDN_CATEGORICAL = [
    "id_12", "id_15", "id_16", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34",
    "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]


def _load_transaction(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _load_identity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Normalise test_identity column names: id-XX → id_XX
    df.columns = [c.replace("-", "_") if c.startswith("id-") else c for c in df.columns]
    return df


def _join(df_txn: pd.DataFrame, df_idn: pd.DataFrame, id_col: str) -> pd.DataFrame:
    return df_txn.merge(df_idn, on=id_col, how="left")


def _derive_feature_types(X: pd.DataFrame) -> dict:
    all_cat = [c for c in TXN_CATEGORICAL + IDN_CATEGORICAL if c in X.columns]
    all_num = [c for c in X.select_dtypes(include=np.number).columns if c not in all_cat]
    return {
        "numerical":   all_num,
        "categorical": all_cat,
        "temporal":    ["TransactionDT"] if "TransactionDT" in X.columns else [],
    }


# ── public API ────────────────────────────────────────────────────────────────

def load_ieee_cis(
    raw_path: str,
    config: dict = None,
    split: str = "train",
) -> tuple:
    """Load IEEE-CIS dataset.

    Args:
        raw_path: Directory containing the IEEE-CIS CSV files.
        config:   Optional dataset config dict from datasets.yaml.
        split:    'train'   → labeled data only  (returns X, y, feature_types)
                  'test'    → unlabeled data only (returns X, feature_types)
                  'all'     → train + test concatenated, label column kept where
                              available and set to NaN for test rows
                              (returns X, y, feature_types)

    Returns:
        split='train' or 'all': (X, y, feature_types)
        split='test':           (X, feature_types)
    """
    raw_dir  = Path(raw_path)
    label_col = config["label_col"]  if config else "isFraud"
    id_col    = config.get("id_col", "TransactionID") if config else "TransactionID"
    drop_cols = (config.get("drop_cols") or []) if config else []

    if split not in ("train", "test", "all"):
        raise ValueError(f"split must be 'train', 'test', or 'all', got: '{split}'")

    frames = []

    if split in ("train", "all"):
        txn = _load_transaction(raw_dir / "train_transaction.csv")
        idn_path = raw_dir / "train_identity.csv"
        if idn_path.exists():
            idn = _load_identity(idn_path)
            txn = _join(txn, idn, id_col)
        frames.append(txn)

    if split in ("test", "all"):
        txn = _load_transaction(raw_dir / "test_transaction.csv")
        idn_path = raw_dir / "test_identity.csv"
        if idn_path.exists():
            idn = _load_identity(idn_path)
            txn = _join(txn, idn, id_col)
        # Test rows have no label — add NaN column so concat works
        if split == "all":
            txn[label_col] = np.nan
        frames.append(txn)

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    cols_to_drop = [id_col] + [c for c in drop_cols if c in df.columns]

    if split == "test":
        X = df.drop(columns=cols_to_drop, errors="ignore")
        return X, _derive_feature_types(X)

    # train or all — extract label
    y = df[label_col]
    X = df.drop(columns=[label_col] + cols_to_drop, errors="ignore")

    if split == "train":
        y = y.astype(int)
    # split='all': y contains float with NaN for test rows — caller decides how to use

    return X, y, _derive_feature_types(X)
