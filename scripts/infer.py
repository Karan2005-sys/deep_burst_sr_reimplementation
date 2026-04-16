from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dbsr.data.io import pack_raw_bayer, read_image, to_float_tensor
from dbsr.models.dbsr import DeepBurstSR
from dbsr.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--burst-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device_name = config["device"] if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    frame_paths = sorted(Path(args.burst_dir).glob(config["dataset"]["frame_pattern"]))[: config["dataset"]["burst_size"]]
    if not frame_paths:
        raise RuntimeError(f"No frames found in {args.burst_dir}")

    burst = []
    for frame_path in frame_paths:
        raw = read_image(frame_path)
        raw_t = to_float_tensor(raw, config["dataset"]["normalize_divisor"])
        burst.append(pack_raw_bayer(raw_t))
    burst_t = torch.stack(burst, dim=0).unsqueeze(0).to(device)

    model = DeepBurstSR(**config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pred, _ = model(burst_t)
    image = pred[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    image_u8 = (image * 255.0).astype("uint8")
    cv2.imwrite(args.output, cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
