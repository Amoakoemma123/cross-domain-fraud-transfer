"""Feature engineering: domain-agnostic transformations."""

import pandas as pd
import numpy as np


def add_log_amount(X: pd.DataFrame, amount_col: str = "amount") -> pd.DataFrame:
    """Add log1p-transformed amount feature."""
    X = X.copy()
    if amount_col in X.columns:
        X[f"{amount_col}_log"] = np.log1p(X[amount_col].clip(lower=0))
    return X


def add_balance_diff(X: pd.DataFrame) -> pd.DataFrame:
    """Add balance difference features (PaySim-specific)."""
    X = X.copy()
    if "oldbalanceOrg" in X.columns and "newbalanceOrig" in X.columns:
        X["balance_diff_orig"] = X["newbalanceOrig"] - X["oldbalanceOrg"]
    if "oldbalanceDest" in X.columns and "newbalanceDest" in X.columns:
        X["balance_diff_dest"] = X["newbalanceDest"] - X["oldbalanceDest"]
    return X


def add_hour_of_day(X: pd.DataFrame, time_col: str = "Time", dataset: str = "ulb") -> pd.DataFrame:
    """Derive hour-of-day from a temporal column.

    ULB     'Time' is seconds elapsed from first transaction (// 3600 % 24).
    PaySim  'step' is hours (1 step = 1 hour, % 24).
    BankSim 'step' is simulation steps treated as hours (% 24).
    IEEE-CIS 'TransactionDT' is seconds offset from reference date.
    """
    X = X.copy()
    if dataset == "ulb" and "Time" in X.columns:
        X["hour_of_day"] = (X["Time"] // 3600) % 24
    elif dataset in ("paysim", "banksim") and "step" in X.columns:
        X["hour_of_day"] = X["step"] % 24
    elif dataset == "ieee_cis" and "TransactionDT" in X.columns:
        X["hour_of_day"] = (X["TransactionDT"] // 3600) % 24
    return X


def engineer_features(X: pd.DataFrame, dataset: str, config: dict = None) -> pd.DataFrame:
    """Apply all relevant feature engineering for a dataset."""
    cfg = config or {}
    amount_col = {"ulb": "Amount", "ieee_cis": "TransactionAmt",
                  "paysim": "amount", "banksim": "amount"}.get(dataset, "amount")

    if cfg.get("amount_log", True):
        X = add_log_amount(X, amount_col)
    if cfg.get("balance_diff", True) and dataset == "paysim":
        X = add_balance_diff(X)

    X = add_hour_of_day(X, dataset=dataset)
    return X
