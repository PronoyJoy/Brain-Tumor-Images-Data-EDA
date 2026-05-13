# Phase 1 — Exploratory Data Analysis

## Goal
Understand the dataset before touching it. Class balance, image dimensions, channel structure, pixel-intensity distributions, and sanity checks.

## How to reproduce
```bash
python src/eda.py
```

## 1.1 Class distribution

| Class | Training | Testing |
|---|---|---|
| glioma | 1400 | 400 |
| meningioma | 1400 | 400 |
| notumor | 1400 | 400 |
| pituitary | 1400 | 400 |

- Imbalance ratio: 1.00 (perfectly balanced).
- **However**, the original Kaggle dataset is normally uneven. This balance was achieved by duplication, not augmentation. See `DUPLICATES.md`.

Figure: `outputs/figures/class_distribution.png`

## 1.2 Image dimension audit
Random sample of 200 images per (class, split) — 1,600 images total.

| | width | height | channels | file_size_kb |
|---|---|---|---|---|
| min | 150 | 168 | 1 | 4.3 |
| mean | 449 | 453 | 2.1 | 22.2 |
| max | 1335 | 1427 | 3 | 261.8 |

- **67 %** of sampled images are exactly **512 × 512**.
- Other shapes: 225×225 (4.7 %), 630×630, 201×251, 227×222, plus a long tail.
- **Channel mix:** 897 RGB (56 %) + 703 grayscale (44 %). The brief's claim that everything is 3-channel JPG is wrong — the pipeline must coerce.

Figure: `outputs/figures/dimension_scatter.png`
CSV: `outputs/processed/dim_audit.csv`

## 1.3 Pixel intensity per class
150 Training images per class, converted to grayscale (0–255):

| Class | mean of per-image means | mean of per-image stds |
|---|---|---|
| glioma | 31.67 | 37.53 |
| meningioma | 43.61 | 46.66 |
| **notumor** | **61.05** | **60.31** |
| pituitary | 49.04 | 41.66 |

`notumor` is roughly 2× brighter than `glioma`. Samplewise standardization in Phase 2 neutralizes this for training.

Figure: `outputs/figures/pixel_intensity_per_class.png`

## 1.4 Sample grid
4 random images per class, displayed grayscale with their native dimensions.

Figure: `outputs/figures/sample_grid.png`

## 1.5 Sanity checks
| Check | Result |
|---|---|
| Corrupted images | 0 |
| Non-image files | 0 |
| Cross-class filename collisions | 0 |
| Content-hash duplicate groups (fast hash) | 153 |

The fast-hash duplicates were verified with full MD5 — full results in `DUPLICATES.md`.

## Files added
- `src/eda.py` — single-script EDA producing all four figures and CSV.
