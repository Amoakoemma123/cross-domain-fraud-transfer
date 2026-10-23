"""Multilayer Perceptron (PyTorch) for fraud detection."""

import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (256, 128, 64),
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
