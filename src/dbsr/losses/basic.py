from __future__ import annotations

import torch
from torch import nn


class BasicL1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> dict:
        return {
            "loss": (pred - gt).abs().mean(),
            "aligned_pred": pred,
        }
