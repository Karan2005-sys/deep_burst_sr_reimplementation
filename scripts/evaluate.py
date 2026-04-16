from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dbsr.engine.trainer import Trainer
from dbsr.models.dbsr import DeepBurstSR
from dbsr.utils.builders import build_dataset
from dbsr.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    device_name = config["device"] if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    dataset = build_dataset(config, args.data_root, config["dataset"]["val_split"])
    loader = DataLoader(
        dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )

    model = DeepBurstSR(**config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    trainer = Trainer(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4), config=config, device=device)
    metrics = trainer.validate(loader)
    print(metrics)


if __name__ == "__main__":
    main()
