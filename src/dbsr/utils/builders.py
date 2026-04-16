from __future__ import annotations

from torch.utils.data import Dataset


def build_dataset(config: dict, data_root: str, split: str) -> Dataset:
    dataset_type = config["dataset_type"]
    dataset_cfg = config["dataset"]

    if dataset_type == "burstsr_real":
        from dbsr.data.burstsr import BurstSRDataset

        return BurstSRDataset(
            root=data_root,
            split=split,
            burst_size=dataset_cfg["burst_size"],
            gt_filename=dataset_cfg["gt_filename"],
            frame_pattern=dataset_cfg["frame_pattern"],
            normalize_divisor=dataset_cfg["normalize_divisor"],
        )

    if dataset_type == "synthetic_rgb":
        from dbsr.data.synthetic import SyntheticBurstRGBDataset

        return SyntheticBurstRGBDataset(
            root=data_root,
            split=split,
            burst_size=dataset_cfg["burst_size"],
            crop_size=dataset_cfg["crop_size"],
            scale=dataset_cfg["scale"],
            max_translation=dataset_cfg["max_translation"],
            max_rotation_deg=dataset_cfg["max_rotation_deg"],
            shot_noise=dataset_cfg["shot_noise"],
            read_noise=dataset_cfg["read_noise"],
            deterministic_val=dataset_cfg.get("deterministic_val", True),
        )

    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def build_loss(config: dict):
    task = config["task"]
    if task == "real":
        from dbsr.losses.alignment import RealBurstAlignedLoss

        return RealBurstAlignedLoss(**config["loss"])
    if task == "synthetic":
        from dbsr.losses.basic import BasicL1Loss

        return BasicL1Loss()
    raise ValueError(f"Unknown task: {task}")
