"""Feature space alignment for cross-domain experiments.

Handles the mismatch between datasets with different column sets.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def align_by_intersection(
    X_source: pd.DataFrame, X_target: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only columns present in both source and target."""
    shared = list(X_source.columns.intersection(X_target.columns))
    if not shared:
        raise ValueError("No shared columns between source and target datasets.")
    return X_source[shared], X_target[shared]


def align_by_pca_projection(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
    n_components: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project both datasets into a common PCA space.

    Fit PCA on source; apply same projection to target.
    Used when datasets have incompatible feature spaces.
    """
    pca = PCA(n_components=min(n_components, X_source.shape[1], X_target.shape[1]))
    pca.fit(X_source)

    cols = [f"pca_{i}" for i in range(pca.n_components_)]
    X_src_pca = pd.DataFrame(pca.transform(X_source), columns=cols, index=X_source.index)
    X_tgt_pca = pd.DataFrame(pca.transform(X_target), columns=cols, index=X_target.index)

    return X_src_pca, X_tgt_pca, pca
