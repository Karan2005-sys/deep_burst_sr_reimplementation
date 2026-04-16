from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from .io import pack_raw_bayer, read_image, to_float_tensor


@dataclass
class SamplePaths:
    burst_frames: List[Path]
    gt: Path
    sample_id: str


class BurstSRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        burst_size: int,
        gt_filename: str = "gt.png",
        frame_pattern: str = "*.png",
        normalize_divisor: float = 1023.0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.burst_size = burst_size
        self.gt_filename = gt_filename
        self.frame_pattern = frame_pattern
        self.normalize_divisor = normalize_divisor
        self.samples = self._discover()

        if not self.samples:
            raise RuntimeError(f"No BurstSR samples found under {self.root / self.split}")

    def _discover(self) -> List[SamplePaths]:
        split_dir = self.root / self.split
        sample_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        samples: List[SamplePaths] = []

        for sample_dir in sample_dirs:
            burst_dir = sample_dir / "burst"
            gt_path = sample_dir / self.gt_filename
            if not burst_dir.exists() or not gt_path.exists():
                continue
            frames = sorted(burst_dir.glob(self.frame_pattern))
            if len(frames) < self.burst_size:
                continue
            samples.append(
                SamplePaths(
                    burst_frames=frames[: self.burst_size],
                    gt=gt_path,
                    sample_id=sample_dir.name,
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        packed_frames = []

        for frame_path in sample.burst_frames:
            raw_np = read_image(frame_path)
            raw_tensor = to_float_tensor(raw_np, self.normalize_divisor)
            packed_frames.append(pack_raw_bayer(raw_tensor))

        gt_np = read_image(sample.gt)
        gt_tensor = to_float_tensor(gt_np, self.normalize_divisor)

        burst = torch.stack(packed_frames, dim=0)
        return {
            "burst": burst,
            "gt": gt_tensor,
            "sample_id": sample.sample_id,
        }
