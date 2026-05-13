# FINDINGS — Brain Tumor MRI EDA & Preprocessing

Top-level summary across all four phases. See per-phase docs for detail.

## 1. Class balance
- Counts are exactly 1400 / class (Training) and 400 / class (Testing) — *perfectly* balanced.
- This balance is partly synthetic. **187 byte-identical duplicate files** are spread unevenly across classes (Training/notumor lost 119; Training/glioma lost 0). See `DUPLICATES.md`.

## 2. Image size variation
- Range: **150 × 168 to 1335 × 1427** pixels (≈9× span).
- ~67 % of sampled images are 512 × 512. The rest is a long tail (225², 630², 201×251, …).
- Resize target chosen: **224 × 224** (bilinear, no aspect-ratio preservation).

## 3. Channel structure
- Source is **mixed**: ~56 % three-channel RGB, ~44 % single-channel grayscale. The original brief's assumption ("all 3-channel JPGs") is incorrect.
- Pipeline coerces every input to 3-channel RGB via `PIL.convert("RGB")` so pretrained models remain compatible downstream.

## 4. Pixel intensity per class (Training, grayscale 0–255)
| Class | mean | std |
|---|---|---|
| glioma | 31.7 | 37.5 |
| meningioma | 43.6 | 46.7 |
| **notumor** | **61.1** | 60.3 |
| pituitary | 49.0 | 41.7 |

`notumor` is markedly brighter than the tumor classes. Samplewise standardization neutralizes this per image.

## 5. Pipeline performance (CPU, 32-image smoke test)
- NumPy/PIL path: **6.08 ms/image** (194 ms total).
- tf.data parallel path: **0.79 ms/image** post-warmup (25 ms batch).
- Both well under the brief's "a few seconds for 32 images is off" threshold.

## 6. Anomalies worth knowing
- **187 byte-identical duplicates** (`DUPLICATES.md`). Effective Training set ≈ 5,413, not 5,600.
- **Augmentation prefix is misleading.** `Tr-aug-me_*.jpg` files are exact byte copies of original `Tr-me_*.jpg`, not augmented versions.
- **Train/test leakage** has not yet been verified; recommended before training.

## Where things live
| Path | What it is |
|---|---|
| `src/eda.py` | Phase 1 EDA |
| `src/preprocessing.py` | Phase 2 pipeline |
| `src/verify_preprocessing.py` | Phase 2 verification |
| `src/find_duplicates.py` | Duplicate-file scanner |
| `src/smoke_test.py` | Phase 3 smoke test |
| `outputs/figures/` | All EDA + verification plots |
| `outputs/processed/dim_audit.csv` | Dimension audit raw data |
| `outputs/processed/duplicates.csv` | Per-file duplicate map |

## Recommended next steps (deferred, per brief)
1. Train/test leakage check (MD5 across splits).
2. Optional dedup flag on `build_tf_dataset`.
3. MONAI pipeline comparison.
4. Real data augmentation (rotation, flip, intensity jitter).
5. Baseline CNN.
