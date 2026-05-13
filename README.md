# Brain Tumor MRI — EDA & Preprocessing Pipeline

Exploratory data analysis and a clean image preprocessing pipeline for the [Kaggle Brain Tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Outputs `(224, 224, 3) float32` tensors with per-image samplewise standardization, ready for transfer-learning models.

Project brief: see [`claude.md`](claude.md). Cross-phase summary: see [`FINDINGS.md`](FINDINGS.md).

## Quick start

```bash
# 1. Install deps
pip install tensorflow keras numpy pandas matplotlib seaborn pillow scikit-learn tqdm opencv-python

# 2. Place the Kaggle dataset under data/raw/  (data/raw is gitignored)
#    Expected layout:
#      data/raw/Training/{glioma,meningioma,notumor,pituitary}/
#      data/raw/Testing/{glioma,meningioma,notumor,pituitary}/

# 3. Run, in order:
python src/eda.py                    # Phase 1: figures + dim_audit.csv
python src/find_duplicates.py        # Duplicate scan -> duplicates.csv
python src/verify_preprocessing.py   # Phase 2: 17/17 checks
python src/smoke_test.py             # Phase 3: end-to-end timing
```

## Using the pipeline

```python
from src.preprocessing import preprocess, build_tf_dataset

# Single image -> (224, 224, 3) float32
x = preprocess("data/raw/Training/glioma/Tr-gl_0001.jpg")

# tf.data pipeline for batched training
ds = build_tf_dataset("data/raw/Training", batch_size=32)
for batch_x, batch_y in ds.take(1):
    ...
```

## Key findings at a glance

- **Class counts are perfectly 1400/400 per class** — but ~187 byte-identical duplicate files account for the balance. Effective training set is closer to 5,413 (see [`DUPLICATES.md`](DUPLICATES.md)).
- **Image sizes span 150 px to 1,335 px**; 67 % of sampled images are 512×512.
- **Channels are mixed** (~56 % RGB, ~44 % grayscale). Pipeline coerces all inputs to 3-channel RGB.
- **`notumor` is ~2× brighter** than `glioma` in raw pixel intensity. Samplewise standardization neutralizes this per image.
- **Performance:** 6 ms/image (NumPy/PIL) or <1 ms/image (tf.data, post-warmup) on CPU.

## Repository layout

```
.
├── claude.md                        Original project brief
├── FINDINGS.md                      Cross-phase summary
├── PHASE0_SETUP.md                  Setup notes
├── PHASE1_EDA.md                    EDA details
├── PHASE2_PREPROCESSING.md          Pipeline API + verification
├── PHASE3_SMOKE_TEST.md             Timings + checks
├── DUPLICATES.md                    Byte-identical duplicate investigation
├── src/
│   ├── preprocessing.py             Public pipeline API
│   ├── eda.py                       Phase 1 EDA script
│   ├── find_duplicates.py           Full-MD5 duplicate scanner
│   ├── verify_preprocessing.py      Phase 2 verification harness
│   ├── smoke_test.py                Phase 3 end-to-end timing
│   └── _probe_deps.py               Dependency probe
├── outputs/
│   ├── figures/                     PNGs from EDA + verification
│   └── processed/                   dim_audit.csv, duplicates.csv
└── data/raw/                        (gitignored — fetch from Kaggle)
```

## Status

All four phases complete and verified. Open follow-ups (not started):

1. Train/test leakage check (MD5 across splits).
2. Optional `dedup=True` flag for `build_tf_dataset`.
3. MONAI pipeline comparison.
4. Real data augmentation.
5. Baseline CNN.
