from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def _list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def _load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return pil_to_tensor(image).float() / 255.0


def _sample_crop(image: torch.Tensor, size: int, deterministic: bool) -> torch.Tensor:
    _, height, width = image.shape
    if height < size or width < size:
        scale = max(size / height, size / width)
        image = F.interpolate(image.unsqueeze(0), scale_factor=scale + 1e-3, mode="bilinear", align_corners=False)[0]
        _, height, width = image.shape

    if deterministic:
        top = max((height - size) // 2, 0)
        left = max((width - size) // 2, 0)
    else:
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
    return image[:, top : top + size, left : left + size]


def _affine_matrix(angle_deg: float, tx_px: float, ty_px: float, width: int, height: int) -> torch.Tensor:
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    tx = 2.0 * tx_px / max(width - 1, 1)
    ty = 2.0 * ty_px / max(height - 1, 1)
    return torch.tensor([[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=torch.float32)


def _warp_rgb(image: torch.Tensor, angle_deg: float, tx_px: float, ty_px: float) -> torch.Tensor:
    _, height, width = image.shape
    theta = _affine_matrix(angle_deg, tx_px, ty_px, width, height).unsqueeze(0)
    grid = F.affine_grid(theta, size=(1, 3, height, width), align_corners=True)
    return F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)[0]


def _rgb_to_bayer_raw(image: torch.Tensor) -> torch.Tensor:
    _, height, width = image.shape
    raw = torch.zeros(1, height, width, dtype=image.dtype)
    raw[:, 0::2, 0::2] = image[0:1, 0::2, 0::2]
    raw[:, 0::2, 1::2] = image[1:2, 0::2, 1::2]
    raw[:, 1::2, 0::2] = image[1:2, 1::2, 0::2]
    raw[:, 1::2, 1::2] = image[2:3, 1::2, 1::2]
    return raw


def _pack_raw(raw: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [raw[:, 0::2, 0::2], raw[:, 0::2, 1::2], raw[:, 1::2, 0::2], raw[:, 1::2, 1::2]],
        dim=0,
    )


class SyntheticBurstRGBDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        burst_size: int,
        crop_size: int,
        scale: int,
        max_translation: float,
        max_rotation_deg: float,
        shot_noise: float,
        read_noise: float,
        deterministic_val: bool = True,
    ) -> None:
        self.root = Path(root) / split
        self.burst_size = burst_size
        self.crop_size = crop_size
        self.scale = scale
        self.max_translation = max_translation
        self.max_rotation_deg = max_rotation_deg
        self.shot_noise = shot_noise
        self.read_noise = read_noise
        self.deterministic = split != "train" and deterministic_val
        self.image_paths = _list_images(self.root)
        if not self.image_paths:
            raise RuntimeError(f"No RGB images found under {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image = _load_rgb(self.image_paths[index])
        hr_size = self.crop_size * self.scale
        gt = _sample_crop(image, hr_size, self.deterministic)

        burst = []
        for frame_idx in range(self.burst_size):
            if frame_idx == 0:
                angle = 0.0
                tx = 0.0
                ty = 0.0
            elif self.deterministic:
                angle = ((frame_idx % 3) - 1) * 0.5 * self.max_rotation_deg
                tx = ((frame_idx % 5) - 2) * 0.25 * self.max_translation
                ty = (((frame_idx + 2) % 5) - 2) * 0.25 * self.max_translation
            else:
                angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                tx = random.uniform(-self.max_translation, self.max_translation)
                ty = random.uniform(-self.max_translation, self.max_translation)

            warped = _warp_rgb(gt, angle, tx, ty)
            lr = F.interpolate(
                warped.unsqueeze(0),
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
            )[0]
            raw = _rgb_to_bayer_raw(lr)
            noise = torch.randn_like(raw) * (self.read_noise + self.shot_noise * raw.clamp_min(0.0).sqrt())
            raw = (raw + noise).clamp(0.0, 1.0)
            burst.append(_pack_raw(raw))

        return {
            "burst": torch.stack(burst, dim=0),
            "gt": gt,
            "sample_id": self.image_paths[index].stem,
        }
