"""DANN — Domain-Adversarial Neural Network.

Reference: Ganin et al. (2016), "Domain-Adversarial Training of Neural Networks".
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    """Gradient reversal layer for domain adversarial training."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_)


class DANN(nn.Module):
    """Domain-Adversarial Neural Network.

    Architecture:
        Feature encoder → [Fraud classifier]
                        → [GRL → Domain classifier]
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int] = (256, 128),
        latent_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Shared feature encoder
        enc_layers = []
        in_dim = input_dim
        for h in encoder_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        # Fraud classifier
        self.fraud_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Domain classifier (source=0, target=1)
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, x: torch.Tensor, lambda_: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (fraud_logit, domain_logit)."""
        z = self.encoder(x)
        fraud_out = self.fraud_classifier(z).squeeze(-1)
        domain_out = self.domain_classifier(grad_reverse(z, lambda_)).squeeze(-1)
        return fraud_out, domain_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict_fraud_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(x)
            return torch.sigmoid(self.fraud_classifier(z).squeeze(-1))
