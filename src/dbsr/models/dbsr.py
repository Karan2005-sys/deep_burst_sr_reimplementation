from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvRelu, ResidualStack
from .flow import FarnebackFlowEstimator


def build_grid(flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = flow.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype),
        torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    norm_flow_x = flow[:, 0] / max(width - 1, 1) * 2.0
    norm_flow_y = flow[:, 1] / max(height - 1, 1) * 2.0
    flow_grid = torch.stack([norm_flow_x, norm_flow_y], dim=-1)
    return base_grid + flow_grid


def warp_features(features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    grid = build_grid(flow)
    return F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=True)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, embed_dim: int, depth: int) -> None:
        super().__init__()
        self.stem = ConvRelu(in_channels, base_channels)
        self.body = ResidualStack(base_channels, depth)
        self.proj = nn.Conv2d(base_channels, embed_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.proj(x)


class WeightPredictor(nn.Module):
    def __init__(self, embed_dim: int, proj_dim: int, flow_feature_dim: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(embed_dim, proj_dim, 1)
        self.flow_net = nn.Sequential(
            ConvRelu(2, flow_feature_dim, 3),
            nn.Conv2d(flow_feature_dim, flow_feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            ConvRelu(proj_dim * 2 + flow_feature_dim, embed_dim, 3),
            ResidualStack(embed_dim, 2),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

    def forward(self, aligned: torch.Tensor, base: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        aligned_proj = self.project(aligned)
        base_proj = self.project(base)
        residual = aligned_proj - base_proj
        flow_fraction = flow - torch.floor(flow)
        flow_feat = self.flow_net(flow_fraction)
        features = torch.cat([base_proj, residual, flow_feat], dim=1)
        return self.net(features)


class Decoder(nn.Module):
    def __init__(self, embed_dim: int, lowres_depth: int, highres_depth: int, scale: int) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualStack(128, lowres_depth),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(128, 128 * (scale * 2) ** 2, 3, padding=1),
            nn.PixelShuffle(scale * 2),
            nn.ReLU(inplace=True),
        )
        self.post = nn.Sequential(
            ResidualStack(128, highres_depth),
            nn.Conv2d(128, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.upsample(x)
        return self.post(x)


class DeepBurstSR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        embed_dim: int,
        encoder_blocks: int,
        decoder_blocks_lowres: int,
        decoder_blocks_highres: int,
        fusion_proj_dim: int,
        flow_feature_dim: int,
        scale: int,
        flow_backend: str = "farneback",
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, embed_dim, encoder_blocks)
        self.weight_predictor = WeightPredictor(embed_dim, fusion_proj_dim, flow_feature_dim)
        self.decoder = Decoder(embed_dim, decoder_blocks_lowres, decoder_blocks_highres, scale)

        if flow_backend != "farneback":
            raise ValueError(f"Unsupported flow backend: {flow_backend}")
        self.flow_estimator = FarnebackFlowEstimator()

    def forward(self, burst: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        _, frames, _, _, _ = burst.shape
        encoded = []
        for idx in range(frames):
            encoded.append(self.encoder(burst[:, idx]))
        encoded = torch.stack(encoded, dim=1)

        flows = self.flow_estimator(burst)
        base = encoded[:, 0]

        aligned = []
        weights = []
        for idx in range(frames):
            current = encoded[:, idx]
            flow = flows[:, idx]
            warped = current if idx == 0 else warp_features(current, flow)
            aligned.append(warped)
            weights.append(self.weight_predictor(warped, base, flow))

        aligned_t = torch.stack(aligned, dim=1)
        weights_t = torch.softmax(torch.stack(weights, dim=1), dim=1)
        fused = (weights_t * aligned_t).sum(dim=1)
        output = self.decoder(fused)

        extras = {
            "flows": flows,
            "weights": weights_t,
        }
        return output, extras
