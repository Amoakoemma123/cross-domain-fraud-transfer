"""Contrastive representation learning for domain-invariant fraud features.

Uses supervised contrastive loss: pulls same-class samples together,
pushes different-class samples apart, regardless of domain.
Reference: Khosla et al. (2020), "Supervised Contrastive Learning".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int] = (256, 128),
        latent_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in encoder_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Projection head (used during contrastive training only)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Fraud classifier (attached after contrastive pre-training)
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return F.normalize(self.projector(z), dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(x)
            return torch.sigmoid(self.classifier(z).squeeze(-1))


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss.

    Args:
        embeddings: (N, D) L2-normalized embeddings.
        labels:     (N,) binary class labels.
        temperature: Scaling temperature.

    Returns:
        Scalar loss.
    """
    embeddings = F.normalize(embeddings, dim=-1)
    N = embeddings.size(0)

    sim = torch.matmul(embeddings, embeddings.T) / temperature  # (N, N)
    sim_exp = torch.exp(sim)

    # Mask: same class
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    # Exclude self
    self_mask = torch.eye(N, device=embeddings.device)
    pos_mask = pos_mask - self_mask

    denom = sim_exp * (1 - self_mask)
    denom = denom.sum(dim=1, keepdim=True)

    log_prob = sim - torch.log(denom + 1e-8)
    n_pos = pos_mask.sum(dim=1)
    loss = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)

    return loss[n_pos > 0].mean()
