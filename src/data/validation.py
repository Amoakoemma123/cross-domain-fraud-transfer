"""Dataset validation and audit utilities."""

import pandas as pd
import numpy as np
from typing import Optional


def audit_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    feature_types: dict = None,
) -> dict:
    """Generate a full audit report for a loaded dataset.

    Returns a dict suitable for JSON logging or printed output.
    """
    n_rows, n_cols = X.shape
    n_fraud  = int(y.sum())
    n_legit  = int((y == 0).sum())
    fraud_rate = n_fraud / len(y)

    missing_by_col = X.isnull().mean()
    cols_with_missing = int((missing_by_col > 0).sum())
    max_missing = float(missing_by_col.max())
    high_missing_cols = missing_by_col[missing_by_col > 0.50].index.tolist()

    num_cols = (feature_types or {}).get("numerical") or \
               X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = (feature_types or {}).get("categorical") or \
               X.select_dtypes(include="object").columns.tolist()
    cat_cols_present = [c for c in cat_cols if c in X.columns]

    # Per-category cardinality
    cardinality = {}
    for col in cat_cols_present:
        cardinality[col] = int(X[col].nunique())

    # Amount statistics (best-guess amount column)
    amt_col = next((c for c in ["Amount", "TransactionAmt", "amount"] if c in X.columns), None)
    amount_stats = {}
    if amt_col:
        s = X[amt_col].dropna()
        amount_stats = {
            "col": amt_col,
            "mean":   round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std":    round(float(s.std()), 4),
            "min":    round(float(s.min()), 4),
            "max":    round(float(s.max()), 4),
            "p99":    round(float(s.quantile(0.99)), 4),
        }

    report = {
        "dataset":            dataset_name,
        "n_rows":             n_rows,
        "n_cols":             n_cols,
        "n_fraud":            n_fraud,
        "n_legit":            n_legit,
        "fraud_rate":         round(fraud_rate, 6),
        "imbalance_ratio":    round(n_legit / max(n_fraud, 1), 2),
        "n_numerical":        len([c for c in num_cols if c in X.columns]),
        "n_categorical":      len(cat_cols_present),
        "categorical_cols":   cat_cols_present,
        "cardinality":        cardinality,
        "cols_with_missing":  cols_with_missing,
        "max_missing_rate":   round(max_missing, 4),
        "high_missing_cols":  high_missing_cols,  # >50% missing
        "amount_stats":       amount_stats,
    }
    return report


def print_audit(report: dict) -> None:
    """Pretty-print an audit report."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Dataset Audit: {report['dataset']}")
    print(sep)
    print(f"  Rows              : {report['n_rows']:,}")
    print(f"  Features          : {report['n_cols']}")
    print(f"  Fraud             : {report['n_fraud']:,}  ({report['fraud_rate']*100:.4f}%)")
    print(f"  Legitimate        : {report['n_legit']:,}")
    print(f"  Imbalance ratio   : 1 : {report['imbalance_ratio']:.1f}")
    print(f"  Numerical cols    : {report['n_numerical']}")
    print(f"  Categorical cols  : {report['n_categorical']}  {report['categorical_cols']}")
    if report["cardinality"]:
        card_str = ", ".join(f"{k}={v}" for k, v in report["cardinality"].items())
        print(f"  Cardinality       : {card_str}")
    print(f"  Missing cols      : {report['cols_with_missing']}  (max {report['max_missing_rate']*100:.1f}%)")
    if report["high_missing_cols"]:
        print(f"  High-missing (>50%): {report['high_missing_cols']}")
    if report["amount_stats"]:
        s = report["amount_stats"]
        print(f"  Amount ({s['col']})  : mean={s['mean']:.2f}  median={s['median']:.2f}"
              f"  std={s['std']:.2f}  max={s['max']:.2f}")
    print(sep)
