"""Sklearn-compatible MLP classifier wrapping the PyTorch Trainer."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.models.deep.mlp import MLP
from src.training.trainer import Trainer


class MLPClassifier:
    """Sklearn-style interface for the PyTorch MLP.

    Args:
        input_dim:    Number of input features (set after preprocessing).
        hidden_dims:  Tuple of hidden layer widths.
        dropout:      Dropout rate.
        batch_norm:   Use batch normalisation.
        lr:           Adam learning rate.
        batch_size:   Mini-batch size.
        epochs:       Maximum training epochs.
        patience:     Early-stopping patience (on val loss).
        val_fraction: Fraction of training data held out for early stopping
                      when no explicit val set is provided to fit().
        pos_weight:   Positive-class weight for BCEWithLogitsLoss.
                      If None, computed from y as n_neg/n_pos.
        device:       'cuda', 'cpu', or None (auto-detect).
        random_state: Seed for weight init and val split.
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.3,
        batch_norm: bool = True,
        lr: float = 1e-3,
        batch_size: int = 512,
        epochs: int = 50,
        patience: int = 10,
        val_fraction: float = 0.10,
        pos_weight: float = None,
        device: str = None,
        random_state: int = 42,
    ):
        self.input_dim    = input_dim
        self.hidden_dims  = hidden_dims
        self.dropout      = dropout
        self.batch_norm   = batch_norm
        self.lr           = lr
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.patience     = patience
        self.val_fraction = val_fraction
        self.pos_weight   = pos_weight
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

        self._model: MLP = None
        self._trainer: Trainer = None
        self.history_: list[dict] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> "MLPClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = X.shape[1] if self.input_dim is None else self.input_dim

        # Class-weighted loss
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pw = self.pos_weight if self.pos_weight is not None else (n_neg / max(n_pos, 1))
        pos_weight_tensor = torch.tensor([pw], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Internal val split if not provided
        if X_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.val_fraction,
                stratify=y, random_state=self.random_state,
            )

        self._model = MLP(input_dim, list(self.hidden_dims), self.dropout, self.batch_norm)
        optimizer   = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._trainer = Trainer(
            self._model, optimizer, loss_fn,
            device=self.device, patience=self.patience,
        )
        self.history_ = self._trainer.fit(
            X.astype(np.float32), y.astype(np.float32),
            X_val.astype(np.float32), y_val.astype(np.float32),
            epochs=self.epochs, batch_size=self.batch_size,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, 2) probability array (sklearn convention)."""
        probs_pos = self._trainer.predict_proba(X.astype(np.float32))
        return np.column_stack([1 - probs_pos, probs_pos])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
