# Synthetic Dataset Prep

## Recommended source

Use DIV2K as the default RGB source for synthetic burst generation.

Why DIV2K:

- It is a standard super-resolution benchmark
- It has 800 training images and 100 validation images
- It is easy to explain to an evaluator
- It works well as a clean HR image source for synthetic burst creation

Verified source:

- Hugging Face dataset card: [eugenesiow/Div2k](https://huggingface.co/datasets/eugenesiow/Div2k)

The dataset card shows:

- `train`: 800 rows
- `validation`: 100 rows
- HR images backed by the official DIV2K zips

## One-command preparation

```bash
python deep_burst_sr_reimpl/scripts/prepare_div2k_synthetic.py \
  --output-root /path/to/SyntheticRGB
```

This creates:

```text
SyntheticRGB/
  train/
  val/
```

## Colab path

The Colab notebook already uses:

- `SYNTH_ROOT=/content/drive/MyDrive/SyntheticRGB`
- `OUTPUT_ROOT=/content/drive/MyDrive/dbsr_outputs`

So after running the prep cell, the dataset and outputs are both visible in Drive.
