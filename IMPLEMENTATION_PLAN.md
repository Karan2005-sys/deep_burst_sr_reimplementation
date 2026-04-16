# Implementation Plan

## Goal

Reimplement the CVPR 2021 Deep Burst Super-Resolution pipeline and make progress immediately using a synthetic training path while waiting for the real BurstSR dataset link.

## Dataset status

- The real BurstSR dataset is officially referenced by the authors, but the original public host currently returns 404.
- To avoid blocking, this project now supports synthetic burst generation from ordinary RGB images.
- Once the real BurstSR dataset becomes available, the same model and training scripts can switch back using the real-data config.

## Immediate training strategy

### Step 1: Train on synthetic data

- Prepare an RGB dataset with `train/` and `val/` folders.
- Random HR crops are sampled from each RGB image.
- Each crop is warped into a burst with small translations and rotations.
- The warped views are downsampled and mosaicked into Bayer RAW.
- Shot noise and read noise are added.
- The model learns to reconstruct the original HR RGB crop.

### Step 2: Fine-tune on real BurstSR later

- Replace synthetic config with the real BurstSR config.
- Keep the same model entrypoints.
- Optionally swap the default flow backend with PWC-Net for closer paper fidelity.

## Added project support

- Synthetic RGB burst dataset loader
- Synthetic config file
- Unified dataset/loss builder so train and eval scripts work for both real and synthetic modes
- Colab notebook ready for Drive mount, dependency install, dataset path setup, and checkpoint output

## Suggested path

1. Start with synthetic training in Colab.
2. Check that the network runs, loss drops, and predictions look sensible.
3. Keep collecting the real BurstSR link from the authors or repo issues.
4. Fine-tune on real data when it becomes available.
