from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dbsr.models.dbsr import build_grid


def _blur_tensor(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(image.shape[1], 1, 1, 1)
    return F.conv2d(image, kernel, padding=radius, groups=image.shape[1])


def _estimate_flow(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_np = (pred.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    gt_np = (gt.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    pred_gray = cv2.cvtColor(pred_np, cv2.COLOR_RGB2GRAY)
    gt_gray = cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        pred_gray,
        gt_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return torch.from_numpy(flow).permute(2, 0, 1).to(device=pred.device, dtype=pred.dtype)


def _solve_color_matrix(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = source.permute(1, 2, 0).reshape(-1, 3)
    y = target.permute(1, 2, 0).reshape(-1, 3)
    solution = torch.linalg.lstsq(x, y).solution
    return solution


def _apply_color_matrix(image: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    flat = image.permute(0, 2, 3, 1).reshape(-1, 3)
    corrected = flat @ matrix
    return corrected.reshape(image.shape[0], image.shape[2], image.shape[3], 3).permute(0, 3, 1, 2)


class RealBurstAlignedLoss(nn.Module):
    def __init__(self, gaussian_kernel: int, gaussian_sigma: float, residual_mask_threshold: float) -> None:
        super().__init__()
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
        self.residual_mask_threshold = residual_mask_threshold

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> dict:
        batch = pred.shape[0]
        losses = []
        aligned_preds = []

        for idx in range(batch):
            flow = _estimate_flow(pred[idx], gt[idx]).unsqueeze(0)
            warped = F.grid_sample(
                pred[idx : idx + 1],
                build_grid(flow),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

            gt_blur = _blur_tensor(gt[idx : idx + 1], self.gaussian_kernel, self.gaussian_sigma)
            pred_blur = _blur_tensor(warped, self.gaussian_kernel, self.gaussian_sigma)
            color_matrix = _solve_color_matrix(pred_blur[0], gt_blur[0])
            corrected = _apply_color_matrix(warped, color_matrix)

            residual = (gt_blur - _apply_color_matrix(pred_blur, color_matrix)).pow(2).mean(dim=1, keepdim=True)
            mask = (residual < self.residual_mask_threshold).float()
            mask = F.interpolate(mask, size=corrected.shape[-2:], mode="bilinear", align_corners=False)

            l1 = (mask * (corrected - gt[idx : idx + 1]).abs()).sum() / mask.sum().clamp_min(1.0)
            losses.append(l1)
            aligned_preds.append(corrected)

        total = torch.stack(losses).mean()
        return {
            "loss": total,
            "aligned_pred": torch.cat(aligned_preds, dim=0),
        }
