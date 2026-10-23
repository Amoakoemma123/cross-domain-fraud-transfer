"""Shared MLP encoder backbone for all deep transfer methods.

Used by: CORAL-deep, MMD, Contrastive (pre-training phase), Fine-tune.
DANN uses its own self-contained module (dann.py).
"""

import numpy as np
import torch
import torch.nn as nn


class FraudEncoder(nn.Module):
    """MLP encoder + fraud classifier head.

    Args:
        input_dim:    Number of input features.
        hidden_dims:  Hidden layer widths for the encoder trunk.
        repr_dim:     Dimensionality of the representation (encoder output).
        dropout:      Dropout rate applied after every hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (128, 64),
        repr_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, repr_dim))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

        # Fraud classifier head (binary)
        self.head = nn.Linear(repr_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return repr_dim representation."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud logit (scalar per sample)."""
        return self.head(self.encode(x)).squeeze(-1)

    def predict_proba(self, X: np.ndarray, device: str = "cpu", batch_size: int = 2048) -> np.ndarray:
        """Return (N,) positive-class probabilities."""
        self.eval()
        probs = []
        X_t = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                logit = self.forward(X_t[i : i + batch_size])
                probs.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(probs)
