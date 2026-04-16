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
from dbsr.utils.misc import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    set_seed(int(config["seed"]))

    device_name = config["device"] if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    train_ds = build_dataset(config, args.data_root, config["dataset"]["train_split"])
    val_ds = build_dataset(config, args.data_root, config["dataset"]["val_split"])

    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )

    model = DeepBurstSR(**config["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    trainer = Trainer(model=model, optimizer=optimizer, config=config, device=device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
