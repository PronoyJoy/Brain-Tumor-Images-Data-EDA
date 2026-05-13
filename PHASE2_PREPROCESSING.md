# Phase 2 — Image Preprocessing Pipeline

## Goal
Every input image leaves the pipeline as a clean, standardized `(224, 224, 3) float32` tensor with per-image zero mean and unit variance.

## How to reproduce
```bash
python src/verify_preprocessing.py
```

## Public API (`src/preprocessing.py`)

| Function | In | Out | Notes |
|---|---|---|---|
| `load_image(path)` | path | `(H, W, 3) uint8` | PIL `convert("RGB")`; handles both 1-ch and 3-ch input |
| `resize_image(arr, size)` | array | `(size_h, size_w, 3) uint8` | bilinear, no aspect-ratio preservation |
| `samplewise_standardize(arr)` | array | float32 | `(x − mean) / (std + 1e-8)`, per image |
| `preprocess(path, size)` | path | `(224, 224, 3) float32` | composes the three above |
| `build_tf_dataset(dir, batch_size, image_size)` | directory | `tf.data.Dataset` | parallel decode + map-based standardization |

## Design decisions
- **Channels coerced to 3.** Source is mixed grayscale + RGB; we replicate grayscale across all three channels via `PIL.Image.convert("RGB")` so the same tensors can later feed pretrained 3-channel models.
- **Resize ignores aspect ratio.** Per brief. Most images are 512×512 so distortion is limited but not zero.
- **Samplewise standardization** uses stats across H × W × C (single mean/std per image). Channel relationships are preserved.
- **`build_tf_dataset` uses `image_dataset_from_directory`** with `color_mode="rgb"` for consistency with the NumPy path.

## Verification results (`src/verify_preprocessing.py`)

**17 / 17 checks PASS.**

| Check | Detail |
|---|---|
| NumPy/PIL path × 4 classes | shape `(224,224,3)`, dtype `float32`, finite, mean = 0.0000, std = 1.0000 |
| tf.data 32-batch shape & dtype | `(32, 224, 224, 3) float32` |
| tf.data batch has no NaN/Inf | confirmed |
| Per-image mean across batch ≈ 0 | max abs deviation = 1.07 × 10⁻⁷ |
| Per-image std across batch ≈ 1 | max abs deviation = 2.98 × 10⁻⁷ |
| All 4 class labels present in batch | confirmed |
| Before/after figure written | `outputs/figures/preprocessing_before_after.png` |

## Files added
- `src/preprocessing.py`
- `src/verify_preprocessing.py`
- `outputs/figures/preprocessing_before_after.png`
