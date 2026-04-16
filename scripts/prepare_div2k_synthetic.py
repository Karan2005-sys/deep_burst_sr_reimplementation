from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import load_dataset


def _copy_path(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def export_split(output_root: Path, split_name: str, max_items: int | None) -> None:
    dataset = load_dataset("eugenesiow/Div2k", "bicubic_x4", split=split_name)
    split_dir = output_root / ("train" if split_name == "train" else "val")
    split_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset) if max_items is None else min(len(dataset), max_items)
    for idx in range(total):
        record = dataset[idx]
        hr_path = record["hr"]
        target_name = f"{idx:04d}{Path(hr_path).suffix}"
        _copy_path(hr_path, split_dir / target_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    export_split(output_root, "train", args.max_train)
    export_split(output_root, "validation", args.max_val)
    print(f"Prepared synthetic RGB dataset at {output_root}")


if __name__ == "__main__":
    main()
