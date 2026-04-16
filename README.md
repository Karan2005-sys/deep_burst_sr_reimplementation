# Deep Burst Super-Resolution Reimplementation

This folder contains a practical PyTorch reimplementation of the CVPR 2021 paper "Deep Burst Super-Resolution".

It now supports two paths:

- `real`: training on the open BurstSR real-world dataset when the dataset becomes available
- `synthetic`: immediate training from standard RGB images by generating synthetic RAW bursts on the fly

## What is implemented

- Packed Bayer RAW input pipeline
- Paper-style encoder, alignment, fusion, and decoder network
- Real-data aligned loss with masking and color correction
- Synthetic-data training pipeline from RGB images
- Unified train, evaluation, and inference scripts
- Colab notebook with Google Drive mounting and checkpoint output

## Current recommendation

Use the synthetic path first because the original public BurstSR download host is currently unavailable.
The default public RGB source for this project is DIV2K.

## Synthetic dataset layout

Put any ordinary RGB images into a folder like:

```text
SyntheticRGB/
  train/
    image_0001.jpg
    image_0002.jpg
    ...
  val/
    image_1001.jpg
    image_1002.jpg
    ...
```

The loader will:

- sample HR crops
- create small inter-frame motion
- downsample to low-resolution views
- convert to Bayer RAW
- pack RAW to 4 channels
- train the network to reconstruct the HR RGB crop

If you do not already have RGB images, use the included DIV2K prep script:

```bash
python deep_burst_sr_reimpl/scripts/prepare_div2k_synthetic.py ^
  --output-root /path/to/SyntheticRGB
```

## Real BurstSR layout

When you obtain the real dataset, the expected structure is still:

```text
BurstSR/
  train/
    0000/
      burst/
        0000.png
        0001.png
        ...
      gt.png
  val/
    0000/
      burst/
        0000.png
        ...
      gt.png
```

## Quick start on synthetic data

### 1. Install dependencies

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r deep_burst_sr_reimpl/requirements.txt
```

### 2. Train on synthetic RGB images

```bash
python deep_burst_sr_reimpl/scripts/train.py ^
  --config deep_burst_sr_reimpl/configs/synthetic_rgb.yaml ^
  --data-root /path/to/SyntheticRGB ^
  --output-dir /path/to/checkpoints
```

### 3. Evaluate

```bash
python deep_burst_sr_reimpl/scripts/evaluate.py ^
  --config deep_burst_sr_reimpl/configs/synthetic_rgb.yaml ^
  --data-root /path/to/SyntheticRGB ^
  --checkpoint /path/to/checkpoint.pt
```

### 4. Later fine-tune on real BurstSR

```bash
python deep_burst_sr_reimpl/scripts/train.py ^
  --config deep_burst_sr_reimpl/configs/real_burstsr.yaml ^
  --data-root /path/to/BurstSR ^
  --output-dir /path/to/checkpoints
```

## Main files

- `scripts/prepare_div2k_synthetic.py`: builds `SyntheticRGB/train` and `SyntheticRGB/val` from DIV2K
- `configs/synthetic_rgb.yaml`: immediate-start synthetic training config
- `configs/real_burstsr.yaml`: real-data config for later fine-tuning
- `src/dbsr/data/synthetic.py`: on-the-fly synthetic burst generation
- `src/dbsr/utils/builders.py`: shared dataset and loss builders
- `notebooks/deep_burst_sr_burstsr_colab.ipynb`: Colab starter notebook
