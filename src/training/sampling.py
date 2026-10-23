"""Class imbalance sampling strategies."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def apply_sampling(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "class_weight",
    smote_k: int = 5,
    smote_max_per_class: int = 50000,
    undersample_ratio: float = 0.10,
    random_state: int = 42,
) -> tuple:
    """Apply an imbalance sampling strategy to training data.

    Args:
        X:                  Feature array.
        y:                  Label array.
        strategy:           'none', 'class_weight', 'undersample', 'smote', 'adasyn'.
        smote_k:            SMOTE k-neighbours.
        smote_max_per_class: Cap on synthetic samples per class.
        undersample_ratio:  Target minority fraction for undersampling.
        random_state:       RNG seed.

    Returns:
        (X_resampled, y_resampled, class_weights_dict)
        class_weights_dict is None unless strategy == 'class_weight'.
    """
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    class_weights = None

    if strategy == "none":
        return X, y, class_weights

    if strategy == "class_weight":
        class_weights = {0: 1.0, 1: n_neg / max(n_pos, 1)}
        return X, y, class_weights

    if strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        ratio = {0: int(n_pos / max(undersample_ratio, 1e-6)), 1: n_pos}
        sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=random_state)
        X, y = sampler.fit_resample(X, y)
        return X, y, class_weights

    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        target_n = min(n_neg, smote_max_per_class)
        sampling = {1: target_n} if target_n > n_pos else "auto"
        sampler = SMOTE(sampling_strategy=sampling, k_neighbors=smote_k, random_state=random_state)
        X, y = sampler.fit_resample(X, y)
        return X, y, class_weights

    if strategy == "adasyn":
        from imblearn.over_sampling import ADASYN
        target_n = min(n_neg, smote_max_per_class)
        sampling = {1: target_n} if target_n > n_pos else "auto"
        sampler = ADASYN(sampling_strategy=sampling, n_neighbors=smote_k, random_state=random_state)
        X, y = sampler.fit_resample(X, y)
        return X, y, class_weights

    raise ValueError(f"Unknown sampling strategy: {strategy}")
