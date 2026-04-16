from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_split(root: Path, split: str) -> list[dict]:
    split_dir = root / split
    records: list[dict] = []
    for sample_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        burst_dir = sample_dir / "burst"
        gt_path = sample_dir / "gt.png"
        frames = sorted(burst_dir.glob("*.png")) if burst_dir.exists() else []
        records.append(
            {
                "sample_id": sample_dir.name,
                "num_frames": len(frames),
                "has_gt": gt_path.exists(),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="burstsr_manifest.json")
    args = parser.parse_args()

    root = Path(args.data_root)
    manifest = {
        "train": inspect_split(root, "train"),
        "val": inspect_split(root, "val"),
    }
    Path(args.output).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {args.output}")


if __name__ == "__main__":
    main()
