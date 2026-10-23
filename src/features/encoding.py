"""Categorical encoding utilities."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def onehot_encode(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    handle_unknown: str = "ignore",
    max_categories: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit OHE on train, transform both train and test."""
    if not cat_cols:
        return X_train, X_test

    enc = OneHotEncoder(
        handle_unknown=handle_unknown,
        sparse_output=False,
        max_categories=max_categories,
    )

    enc.fit(X_train[cat_cols])

    def _transform(X, cols):
        ohe_arr = enc.transform(X[cols])
        ohe_cols = enc.get_feature_names_out(cols)
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=X.index)
        return pd.concat([X.drop(columns=cols), ohe_df], axis=1)

    return _transform(X_train, cat_cols), _transform(X_test, cat_cols)


def label_encode(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Label-encode categorical columns in-place (fit on X only)."""
    X = X.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X
