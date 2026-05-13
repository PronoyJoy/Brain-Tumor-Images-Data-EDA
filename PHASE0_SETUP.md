# Phase 0 — Setup

## Goal
Stand up the project skeleton, install dependencies, and place the Kaggle Brain Tumor MRI dataset on disk.

## Folder structure created
```
Data Processing Brain MRI/
├── claude.md
├── data/raw/
│   ├── Training/{glioma,meningioma,notumor,pituitary}/
│   └── Testing/{glioma,meningioma,notumor,pituitary}/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── processed/
└── src/
```

## Environment
- Python 3.13.3, pip 25.1.1
- TensorFlow 2.21 + Keras 3.14 installed cleanly on Python 3.13 (Windows). Earlier compatibility concern was outdated.

## Dependencies installed
| Package | Version |
|---|---|
| tensorflow | 2.21.0 |
| keras | 3.14.1 |
| numpy | 2.2.4 |
| pandas | 2.2.3 |
| matplotlib | 3.10.6 |
| seaborn | 0.13.2 |
| pillow | 11.2.1 |
| scikit-learn | 1.8.0 |
| tqdm | 4.67.3 |
| opencv-python | 4.13.0 |
| kaggle (CLI) | 2.1.2 |

## Dataset
- Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Acquired manually by user (Kaggle API credentials not configured).
- 7,200 `.jpg` files in total — 5,600 Training + 1,600 Testing.
- Folder layout matches the brief exactly.

## Files added
- `src/_probe_deps.py` — utility to verify all required packages import.
