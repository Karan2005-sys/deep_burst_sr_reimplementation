# Output readiness and generated artifacts

## Is the project ready for output?

**Partially yes.**

- ✅ The synthetic training pipeline now runs end-to-end in this environment without OpenCV system libraries.
- ✅ A full smoke run completed and artifacts were generated under `outputs/smoke_run/`.
- ⚠️ The original Colab path that prepares DIV2K via Hugging Face may still fail in restricted network/proxy environments.

## What was fixed

1. **Lazy dataset/loss imports** in `src/dbsr/utils/builders.py` so synthetic runs no longer require real-dataset dependencies at import time.
2. **OpenCV-free image saving** in `src/dbsr/utils/misc.py` by using PIL for RGB output.
3. **Optional OpenCV usage** in:
   - `src/dbsr/data/io.py` (only required when reading real BurstSR images)
   - `src/dbsr/models/flow.py` (falls back to zero-flow tensors when OpenCV/GL is unavailable)
4. **Synthetic data scaling fix** in `src/dbsr/data/synthetic.py` to generate LR frames at `crop_size` (instead of `crop_size * 2`) so model output size matches GT.

## Run command used

```bash
python scripts/train.py \
  --config outputs/smoke_run/smoke_synthetic.yaml \
  --data-root /tmp/SyntheticRGB \
  --output-dir outputs/smoke_run
```

## Stored outputs

- `outputs/smoke_run/train.log`
- `outputs/smoke_run/ARTIFACTS.md` (text manifest with output metadata)


## Binary artifact policy

This environment for review does not support committing binary files.

- PNG prediction/target files were generated during smoke run locally but are **not committed**.
- A text manifest with file sizes and SHA256 hashes is committed instead at `outputs/smoke_run/ARTIFACTS.md`.
- Regenerate visuals with the run command in this document whenever needed.
