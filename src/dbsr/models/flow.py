from __future__ import annotations

from typing import List
import importlib.util
import ctypes.util

import numpy as np
import torch
from torch import nn

from dbsr.data.io import packed_raw_to_rgb_proxy

_CV2_SPEC = importlib.util.find_spec("cv2")
_HAS_LIBGL = ctypes.util.find_library("GL") is not None
if _CV2_SPEC is not None and _HAS_LIBGL:
    import cv2
else:
    cv2 = None


def _to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (array * 255.0).astype(np.uint8)


class FarnebackFlowEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, burst: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = burst.shape
        if channels != 4:
            raise ValueError("Flow estimator expects packed RAW input with 4 channels.")

        if cv2 is None:
            return torch.zeros(batch, frames, 2, height, width, dtype=burst.dtype, device=burst.device)

        flows: List[torch.Tensor] = []
        for batch_idx in range(batch):
            base_rgb = packed_raw_to_rgb_proxy(burst[batch_idx, 0])
            base_gray = cv2.cvtColor(_to_uint8_rgb(base_rgb), cv2.COLOR_RGB2GRAY)
            sample_flows = [torch.zeros(2, height, width, dtype=burst.dtype, device=burst.device)]

            for frame_idx in range(1, frames):
                frame_rgb = packed_raw_to_rgb_proxy(burst[batch_idx, frame_idx])
                frame_gray = cv2.cvtColor(_to_uint8_rgb(frame_rgb), cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    frame_gray,
                    base_gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                flow_t = torch.from_numpy(flow).permute(2, 0, 1).to(device=burst.device, dtype=burst.dtype)
                sample_flows.append(flow_t)

            flows.append(torch.stack(sample_flows, dim=0))
        return torch.stack(flows, dim=0)
