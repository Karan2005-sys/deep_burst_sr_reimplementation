from __future__ import annotations

from pathlib import Path
import importlib.util
import ctypes.util

import numpy as np
import torch

_CV2_SPEC = importlib.util.find_spec("cv2")
_HAS_LIBGL = ctypes.util.find_library("GL") is not None
if _CV2_SPEC is not None and _HAS_LIBGL:
    import cv2
else:
    cv2 = None


def read_image(path: str | Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to read real BurstSR images.")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def to_float_tensor(image: np.ndarray, divisor: float) -> torch.Tensor:
    if image.ndim == 2:
        tensor = torch.from_numpy(image.astype(np.float32) / divisor).unsqueeze(0)
    else:
        image = image.astype(np.float32) / divisor
        tensor = torch.from_numpy(image).permute(2, 0, 1)
    return tensor.contiguous()


def pack_raw_bayer(raw: torch.Tensor) -> torch.Tensor:
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"Expected raw tensor of shape [1, H, W], got {tuple(raw.shape)}")
    _, height, width = raw.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("RAW dimensions must be even for Bayer packing.")

    r = raw[:, 0::2, 0::2]
    g1 = raw[:, 0::2, 1::2]
    g2 = raw[:, 1::2, 0::2]
    b = raw[:, 1::2, 1::2]
    return torch.cat([r, g1, g2, b], dim=0)


def packed_raw_to_rgb_proxy(packed_raw: torch.Tensor) -> torch.Tensor:
    if packed_raw.ndim != 3 or packed_raw.shape[0] != 4:
        raise ValueError(f"Expected packed RAW [4, H, W], got {tuple(packed_raw.shape)}")
    r = packed_raw[0:1]
    g = 0.5 * (packed_raw[1:2] + packed_raw[2:3])
    b = packed_raw[3:4]
    return torch.cat([r, g, b], dim=0)
