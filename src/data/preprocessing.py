"""DatasetPreprocessor — unified fit/transform pipeline for all datasets.

Pipeline steps (applied in order, fitted on train only):
  1. Drop columns with missing rate > threshold (default 50%)
  2. Feature engineering  (log-amount, balance_diff, hour_of_day)
  3. Median imputation for numeric columns
  4. Mode imputation + OHE for categorical columns (max_categories cap)
  5. RobustScaler on all resulting numeric features

Usage:
    prep = DatasetPreprocessor(feature_types, dataset_id='ulb')
    X_train_np = prep.fit_transform(X_train_df)
    X_test_np  = prep.transform(X_test_df)
    feature_names = prep.feature_names_out
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, OneHotEncoder

from src.features.engineering import engineer_features


_SCALERS = {
    "robust":   RobustScaler,
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
}


class DatasetPreprocessor:

    def __init__(
        self,
        feature_types: dict,
        dataset_id: str,
        missing_threshold: float = 0.50,
        ohe_max_categories: int = 50,
        scale_method: str = "robust",
    ):
        self.feature_types      = feature_types
        self.dataset_id         = dataset_id
        self.missing_threshold  = missing_threshold
        self.ohe_max_categories = ohe_max_categories
        self.scale_method       = scale_method

        self._num_imputer = SimpleImputer(strategy="median")
        self._cat_imputer = SimpleImputer(strategy="most_frequent")
        self._ohe = OneHotEncoder(
            max_categories=ohe_max_categories,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        )
        self._scaler = _SCALERS[scale_method]()

        # Populated during fit()
        self._cols_to_drop: list[str] = []
        self._num_cols: list[str]     = []
        self._cat_cols: list[str]     = []
        self._has_ohe: bool           = False
        self._feature_names_out: list[str] = []
        self.is_fitted: bool          = False

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame) -> "DatasetPreprocessor":
        X = X_train.copy()

        # 1. Drop high-missing columns (decided on train statistics only)
        missing_rate = X.isnull().mean()
        self._cols_to_drop = missing_rate[missing_rate > self.missing_threshold].index.tolist()
        X = X.drop(columns=self._cols_to_drop)

        # 2. Feature engineering (deterministic, no fitting needed)
        X = engineer_features(X, self.dataset_id)

        # 3. Determine numeric / categorical columns
        declared_cat = self.feature_types.get("categorical", [])
        self._cat_cols = [c for c in declared_cat if c in X.columns]
        self._num_cols = [c for c in X.columns  if c not in self._cat_cols]

        # 4. Fit imputers
        if self._num_cols:
            self._num_imputer.fit(X[self._num_cols])
        if self._cat_cols:
            X_cat = X[self._cat_cols].astype(str)
            self._cat_imputer.fit(X_cat)
            self._ohe.fit(X_cat)
            self._has_ohe = True

        # 5. Fit scaler on the full encoded representation
        X_proc = self._encode(X)
        self._scaler.fit(X_proc)

        # 6. Store output feature names
        self._feature_names_out = self._build_feature_names()

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")
        X = X.copy()
        X = X.drop(columns=[c for c in self._cols_to_drop if c in X.columns])
        X = engineer_features(X, self.dataset_id)
        X_proc = self._encode(X)
        return self._scaler.transform(X_proc)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        self.fit(X_train)
        # Re-use _encode on training data directly (avoids double engineering)
        X = X_train.copy()
        X = X.drop(columns=[c for c in self._cols_to_drop if c in X.columns])
        X = engineer_features(X, self.dataset_id)
        X_proc = self._encode(X)
        return self._scaler.transform(X_proc)

    @property
    def feature_names_out(self) -> list[str]:
        return self._feature_names_out

    @property
    def n_features_out(self) -> int:
        return len(self._feature_names_out)

    # ── internals ─────────────────────────────────────────────────────────────

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        """Impute + OHE and return a dense float numpy array."""
        # Numeric part
        if self._num_cols:
            # Align columns — fill any missing col with NaN
            num_df = X.reindex(columns=self._num_cols)
            num_arr = self._num_imputer.transform(num_df)
        else:
            num_arr = np.zeros((len(X), 0))

        # Categorical part
        if self._has_ohe:
            cat_df = X.reindex(columns=self._cat_cols).astype(str)
            cat_arr = self._ohe.transform(cat_df)
        else:
            cat_arr = np.zeros((len(X), 0))

        if cat_arr.shape[1] > 0:
            return np.hstack([num_arr, cat_arr])
        return num_arr

    def _build_feature_names(self) -> list[str]:
        names = list(self._num_cols)
        if self._has_ohe:
            names.extend(self._ohe.get_feature_names_out(self._cat_cols).tolist())
        return names
