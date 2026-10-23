"""CORAL — Correlation Alignment for domain adaptation.

Minimizes the difference between source and target feature covariances.
Reference: Sun & Saenko (2016), "Deep CORAL: Correlation Alignment for Deep DA".
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute CORAL loss between source and target feature matrices.

    Args:
        source: (n_src, d) feature tensor.
        target: (n_tgt, d) feature tensor.

    Returns:
        Scalar CORAL loss.
    """
    d = source.size(1)

    # Covariance matrices
    src_cov = _covariance(source)
    tgt_cov = _covariance(target)

    loss = (src_cov - tgt_cov).pow(2).sum() / (4 * d * d)
    return loss


def _covariance(X: torch.Tensor) -> torch.Tensor:
    n = X.size(0)
    X_centered = X - X.mean(dim=0, keepdim=True)
    return (X_centered.T @ X_centered) / (n - 1)


class CORALAdapter(BaseEstimator, TransformerMixin):
    """Sklearn-compatible CORAL feature alignment (closed-form, numpy).

    Whitens source features and colors them with target statistics.
    """

    def fit(self, X_source: np.ndarray, X_target: np.ndarray):
        self.src_mean_ = X_source.mean(axis=0)
        self.tgt_mean_ = X_target.mean(axis=0)

        src_centered = X_source - self.src_mean_
        tgt_centered = X_target - self.tgt_mean_

        n_src = X_source.shape[0]
        n_tgt = X_target.shape[0]

        C_src = (src_centered.T @ src_centered) / (n_src - 1) + np.eye(X_source.shape[1])
        C_tgt = (tgt_centered.T @ tgt_centered) / (n_tgt - 1) + np.eye(X_target.shape[1])

        # Whitening transform: C_src^{-1/2}
        U_src, s_src, _ = np.linalg.svd(C_src)
        self.W_src_ = U_src @ np.diag(1.0 / np.sqrt(s_src + 1e-10)) @ U_src.T

        # Coloring transform: C_tgt^{1/2}
        U_tgt, s_tgt, _ = np.linalg.svd(C_tgt)
        self.W_tgt_ = U_tgt @ np.diag(np.sqrt(s_tgt + 1e-10)) @ U_tgt.T

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.src_mean_
        X_whitened = X_centered @ self.W_src_
        X_colored = X_whitened @ self.W_tgt_ + self.tgt_mean_
        return X_colored
