from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_rgb_image(image: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
