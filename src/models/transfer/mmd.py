"""MMD — Maximum Mean Discrepancy loss for domain adaptation.

Reference: Gretton et al. (2012), "A Kernel Two-Sample Test".
"""

import torch
import torch.nn as nn


def mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel: str = "rbf",
    bandwidth: float = None,
) -> torch.Tensor:
    """Compute unbiased MMD^2 between source and target feature tensors.

    Args:
        source:    (n_src, d) tensor.
        target:    (n_tgt, d) tensor.
        kernel:    'rbf' or 'linear'.
        bandwidth: RBF kernel bandwidth; if None, uses median heuristic.

    Returns:
        Scalar MMD^2 loss.
    """
    if kernel == "linear":
        return _linear_mmd(source, target)
    elif kernel == "rbf":
        return _rbf_mmd(source, target, bandwidth)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def _rbf_mmd(source: torch.Tensor, target: torch.Tensor, bandwidth: float = None) -> torch.Tensor:
    n = source.size(0)
    m = target.size(0)

    if bandwidth is None:
        # Median heuristic
        combined = torch.cat([source, target], dim=0)
        dists = torch.cdist(combined, combined)
        bandwidth = dists.median().item() / (2 * (n + m) ** 0.5 + 1e-8)
        bandwidth = max(bandwidth, 1e-2)

    gamma = 1.0 / (2 * bandwidth ** 2)

    K_ss = _rbf_kernel(source, source, gamma)
    K_tt = _rbf_kernel(target, target, gamma)
    K_st = _rbf_kernel(source, target, gamma)

    mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    return mmd.clamp(min=0)


def _rbf_kernel(X: torch.Tensor, Y: torch.Tensor, gamma: float) -> torch.Tensor:
    dists = torch.cdist(X, Y).pow(2)
    return torch.exp(-gamma * dists)


def _linear_mmd(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = source.mean(dim=0) - target.mean(dim=0)
    return (delta ** 2).sum()
