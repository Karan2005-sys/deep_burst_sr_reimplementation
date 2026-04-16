from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dbsr.utils.builders import build_loss
from dbsr.utils.misc import ensure_dir, save_rgb_image


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).item())


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(gt, window, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(gt * gt, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * gt, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float((numerator / denominator.clamp_min(1e-12)).mean().item())


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scaler = GradScaler(enabled=bool(config["train"]["amp"]) and device.type == "cuda")
        self.criterion = build_loss(config)
        self.output_dir = ensure_dir(config["output_dir"])
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["train"]["epochs"]
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("val/psnr", val_metrics["psnr"], epoch)
            self.writer.add_scalar("val/ssim", val_metrics["ssim"], epoch)

            checkpoint_path = self.output_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "config": self.config,
                    "metrics": val_metrics,
                },
                checkpoint_path,
            )

    def _run_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"train {epoch:03d}")

        for step, batch in enumerate(progress, start=1):
            burst = batch["burst"].to(self.device)
            gt = batch["gt"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                pred, _ = self.model(burst)
                loss = self.criterion(pred, gt)["loss"]

            self.scaler.scale(loss).backward()
            if self.config["train"]["grad_clip_norm"] is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["train"]["grad_clip_norm"])
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item()
            progress.set_postfix(loss=f"{running / step:.4f}")

        return running / max(len(loader), 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int | None = None) -> Dict[str, float]:
        self.model.eval()
        running = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        saved_preview = False

        for batch in tqdm(loader, desc="val"):
            burst = batch["burst"].to(self.device)
            gt = batch["gt"].to(self.device)
            pred, _ = self.model(burst)
            loss = self.criterion(pred, gt)["loss"]
            running += loss.item()
            running_psnr += compute_psnr(pred, gt)
            running_ssim += compute_ssim(pred, gt)

            if epoch is not None and not saved_preview:
                preview_dir = ensure_dir(self.output_dir / "visuals" / f"epoch_{epoch:03d}")
                save_rgb_image(pred[0], preview_dir / "prediction.png")
                save_rgb_image(gt[0], preview_dir / "target.png")
                saved_preview = True

        denom = max(len(loader), 1)
        avg = running / denom
        avg_psnr = running_psnr / denom
        avg_ssim = running_ssim / denom
        return {"loss": avg, "psnr": avg_psnr, "ssim": avg_ssim}
