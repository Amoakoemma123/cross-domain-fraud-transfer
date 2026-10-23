"""Autoencoder (PyTorch) for anomaly-based fraud detection."""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int] = (128, 64, 32),
        latent_dim: int = 16,
    ):
        super().__init__()

        # Encoder
        enc_layers = []
        in_dim = input_dim
        for h in encoder_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(encoder_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Mean squared reconstruction error per sample (used as anomaly score)."""
        x_hat, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)
