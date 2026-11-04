"""Inference-only definition of the trained V3 monotonic opacity controller.

The architecture must stay byte-for-byte compatible with checkpoints produced
by Gaussian-Garments/opacity_controller_v3_common.py.  Training/evaluation
helpers are intentionally omitted from the frontend overlay.
"""

from __future__ import annotations

import torch
from torch import nn


K_BINS = 8
SPLINE_EPS = 1e-6


class SplineOpacityController(nn.Module):
    """Eight-bin monotonic CDF controller used by the project report."""

    def __init__(self, bins: int = K_BINS, eps: float = SPLINE_EPS):
        super().__init__()
        self.bins = bins
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, bins),
        )

    def bin_probabilities(self, base_alpha, canonical_xyz_norm):
        logits = self.net(torch.cat((base_alpha, canonical_xyz_norm), dim=-1))
        weights = torch.nn.functional.softplus(logits) + self.eps
        return weights / weights.sum(dim=-1, keepdim=True)

    def exponent(self, probabilities, opacity):
        if not torch.is_tensor(opacity):
            opacity = probabilities.new_tensor(float(opacity))
        opacity = opacity.to(probabilities)
        if opacity.numel() != 1:
            raise ValueError("V3 expects one scalar requested opacity")
        opacity = opacity.reshape(())
        u = opacity * self.bins
        bin_index = torch.floor(u).long().clamp(0, self.bins - 1)
        fraction = u - bin_index.to(u.dtype)
        before = probabilities.cumsum(dim=-1) - probabilities
        gather_index = bin_index.reshape(1, 1).expand(probabilities.shape[0], 1)
        exponent = before.gather(1, gather_index) + fraction * probabilities.gather(1, gather_index)
        exponent = torch.where(opacity <= 0.0, torch.zeros_like(exponent), exponent)
        exponent = torch.where(opacity >= 1.0, torch.ones_like(exponent), exponent)
        return exponent

    def forward(self, base_alpha, canonical_xyz_norm, opacity):
        probabilities = self.bin_probabilities(base_alpha, canonical_xyz_norm)
        exponent = self.exponent(probabilities, opacity)
        safe = base_alpha.clamp(1e-6, 1.0 - 1e-6)
        controlled = -torch.expm1(exponent * torch.log1p(-safe))
        requested = torch.as_tensor(opacity, device=base_alpha.device, dtype=base_alpha.dtype)
        controlled = torch.where(requested <= 0.0, torch.zeros_like(controlled), controlled)
        controlled = torch.where(requested >= 1.0, base_alpha, controlled)
        return controlled.clamp(0.0, 1.0), probabilities, exponent


def load_v3(path, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    controller = SplineOpacityController().to(device)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()
    return controller, checkpoint
