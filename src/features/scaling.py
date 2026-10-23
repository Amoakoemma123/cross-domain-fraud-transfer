"""Feature scaling utilities."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "robust",
    num_cols: list[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fit scaler on train, transform train and test.

    Args:
        X_train:  Training features.
        X_test:   Test features.
        method:   'standard', 'minmax', or 'robust'.
        num_cols: Columns to scale; defaults to all numeric columns.

    Returns:
        Scaled X_train, X_test, and the fitted scaler.
    """
    if method not in _SCALERS:
        raise ValueError(f"Unknown scaling method '{method}'. Choose from {list(_SCALERS)}")

    if num_cols is None:
        num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    scaler = _SCALERS[method]()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, scaler
