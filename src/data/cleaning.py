"""Dataset cleaning utilities: missing values, duplicates, type coercion."""

import pandas as pd
import numpy as np
from typing import Optional


def drop_high_missing(X: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    """Drop columns with missing rate above threshold."""
    missing_rate = X.isnull().mean()
    cols_to_keep = missing_rate[missing_rate <= threshold].index
    dropped = missing_rate[missing_rate > threshold].index.tolist()
    if dropped:
        print(f"[cleaning] Dropped {len(dropped)} high-missing cols: {dropped}")
    return X[cols_to_keep]


def impute_missing(
    X: pd.DataFrame,
    numerical_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> pd.DataFrame:
    """Impute missing values in numerical and categorical columns."""
    X = X.copy()
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include="object").columns

    for col in num_cols:
        if X[col].isnull().any():
            fill = X[col].median() if numerical_strategy == "median" else X[col].mean()
            X[col] = X[col].fillna(fill)

    for col in cat_cols:
        if X[col].isnull().any():
            fill = X[col].mode()[0]
            X[col] = X[col].fillna(fill)

    return X


def remove_duplicates(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Remove exact duplicate rows."""
    df = X.copy()
    df["__label__"] = y.values
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"[cleaning] Removed {before - after} duplicate rows.")
    y_clean = df.pop("__label__")
    return df, y_clean


def coerce_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types to save memory."""
    X = X.copy()
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = pd.to_numeric(X[col], downcast="float")
    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = pd.to_numeric(X[col], downcast="integer")
    return X
