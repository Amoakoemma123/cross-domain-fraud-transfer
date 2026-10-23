"""Feature harmonization across heterogeneous datasets.

Each dataset has its own feature space. This module maps raw dataset columns
to a shared semantic ontology for cross-domain experiments.
"""

import pandas as pd
import numpy as np
from typing import Optional


# Minimum common feature set available across all datasets
COMMON_SEMANTIC_FEATURES = ["amount", "amount_log"]


def extract_common_features(X: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Extract and rename dataset columns to the common semantic feature space.

    Args:
        X:          Preprocessed feature DataFrame for a single dataset.
        dataset_id: One of 'ulb', 'ieee_cis', 'paysim', 'banksim'.

    Returns:
        DataFrame with only the common semantic features present.
    """
    mapping = _get_column_mapping(dataset_id, X.columns.tolist())
    available = {k: v for k, v in mapping.items() if v in X.columns}
    df = X[list(available.values())].rename(columns={v: k for k, v in available.items()})
    return df


def _get_column_mapping(dataset_id: str, available_cols: list[str]) -> dict[str, str]:
    """Return {semantic_name: raw_col_name} for a given dataset."""
    mappings = {
        "ulb": {
            "amount": "Amount",
        },
        "ieee_cis": {
            "amount": "TransactionAmt",
        },
        "paysim": {
            "amount": "amount",
        },
        "banksim": {
            "amount": "amount",
        },
    }
    return mappings.get(dataset_id, {})


def add_derived_common_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add derived common features (e.g. log-amount) after renaming."""
    X = X.copy()
    if "amount" in X.columns:
        X["amount_log"] = np.log1p(X["amount"].clip(lower=0))
    return X


def align_feature_spaces(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
    strategy: str = "intersection",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align source and target to the same column set.

    Args:
        X_source:  Source domain features.
        X_target:  Target domain features.
        strategy:  'intersection' keeps only shared cols;
                   'union' keeps all cols (fills missing with 0).

    Returns:
        Aligned (X_source, X_target).
    """
    if strategy == "intersection":
        shared = X_source.columns.intersection(X_target.columns)
        return X_source[shared], X_target[shared]
    elif strategy == "union":
        X_source, X_target = X_source.align(X_target, join="outer", axis=1, fill_value=0)
        return X_source, X_target
    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")
