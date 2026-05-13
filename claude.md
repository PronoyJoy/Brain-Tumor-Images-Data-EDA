# 🧠 Brain Tumor MRI — EDA & Image Processing

**Project goal:** Explore the Kaggle Brain Tumor MRI dataset and build a clean preprocessing pipeline using **Keras** (with a path to MONAI later).

**Dataset:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

> 💡 Run the phases in order. Each phase is self-contained — you can stop after any phase and come back later.

---

## 🛠️ Phase 0 — Setup

### Folders to create

```
project/
├── data/
│   └── raw/              # unzipped Kaggle data goes here
├── notebooks/
├── outputs/
│   ├── figures/          # EDA plots
│   └── processed/        # standardized arrays
└── src/
```

### Dependencies

Install these in the active environment:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn pillow scikit-learn tqdm opencv-python
```

### Get the data

**Option A — Kaggle API:**
```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/raw --unzip
```

**Option B — manual:** download the zip from Kaggle and extract into `data/raw/`.

### Expected structure (verify first!)

```
data/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

✅ **First task:** print the folder tree and confirm it matches before going further.

---

## 📊 Phase 1 — Exploratory Data Analysis

> Think of EDA as **meeting the dataset before working with it** — like reading the chart before treating the patient.

Create `notebooks/01_eda.ipynb` (or `src/eda.py`) and run these steps:

### 1.1 Class distribution

- Count images per class for **Training** and **Testing** separately.
- Save a bar plot to `outputs/figures/class_distribution.png`.
- Print whether the dataset is balanced.

> ⚠️ Watch for: imbalance between classes. Even small imbalance affects training.

### 1.2 Image dimension audit

For a random sample of 200 images per class:

- Record `width`, `height`, `channels`, `file_size_kb`.
- Build a pandas DataFrame.
- Print: min, max, mean dimensions.
- Plot a scatter of width vs height.

> 💡 MRI images on this dataset are **not uniform size**. You need to know the range before picking a resize target.

### 1.3 Pixel intensity statistics

For each class, compute on grayscale-converted images:

- Per-image mean and std of pixel intensities.
- Overall class mean and class std.
- Plot histograms of pixel intensities per class (overlay them).

Save to `outputs/figures/pixel_intensity_per_class.png`.

> 🎯 This tells you whether classes differ in brightness/contrast distributions — useful intuition before standardization.

### 1.4 Sample visualisations

- Display a 4×4 grid: 4 random images from each class.
- Title each with class name + dimensions.
- Save to `outputs/figures/sample_grid.png`.

### 1.5 Sanity checks

- Are there any **corrupted images**? (Try opening each with PIL, log failures.)
- Any **duplicate filenames** across classes?
- Any **non-image files** sneaking in?

Print a small report at the end of Phase 1 summarizing all findings.

---

## 🖼️ Phase 2 — Image Preprocessing Pipeline

> The goal: every image leaves this pipeline as a clean, standardized tensor ready for a model.

### 2.1 Resize strategy

- Pick a target size: **224×224** (good default for transfer learning).
- Use bilinear interpolation.
- Keep aspect ratio? → No for now (simple resize). Note this trade-off in a comment.

### 2.2 Channel decision

- The dataset is grayscale MRIs but stored as 3-channel JPGs.
- Decision: **keep 3 channels** so we can use pretrained models later.
- Verify channels are identical (R = G = B) on a sample.

### 2.3 Samplewise standardization

For each image, after resize:

```
standardized = (image - image.mean()) / (image.std() + 1e-8)
```

> 🎯 This is the **samplewise standardization** from your AI for Medicine course — each image becomes zero-mean, unit-variance independently. Especially good for medical imaging where global stats can be misleading.

Implement two versions:

1. A pure NumPy/PIL function (for full control + learning).
2. A `tf.data` pipeline version (for batch training later).

### 2.4 Build the pipeline

Create `src/preprocessing.py` with:

- `load_image(path)` → returns NumPy array
- `resize_image(arr, size=(224, 224))`
- `samplewise_standardize(arr)`
- `preprocess(path)` → composes the above
- `build_tf_dataset(directory, batch_size, image_size)` → returns a `tf.data.Dataset`

### 2.5 Verification (don't skip this!)

After processing a batch:

- ✅ Output shape is `(batch, 224, 224, 3)`
- ✅ Per-image mean ≈ 0
- ✅ Per-image std ≈ 1
- ✅ No NaNs anywhere
- ✅ Visualize 4 before/after pairs and save to `outputs/figures/preprocessing_before_after.png`

Print a clear PASS/FAIL summary.

---

## 🧪 Phase 3 — Quick Validation

Create a tiny script `src/smoke_test.py`:

- Load 32 images using the pipeline.
- Time how long preprocessing takes per image.
- Confirm everything chains together end-to-end.

> 💡 If this takes more than a few seconds for 32 images, something's off — flag it.

---

## 📝 Output Expectations

At the end, the repo should contain:

- `outputs/figures/` → 4-5 EDA + verification plots
- `src/preprocessing.py` → reusable pipeline
- `notebooks/01_eda.ipynb` → EDA narrative (or `src/eda.py` log output)
- A short **`FINDINGS.md`** summarizing:
  - Class balance
  - Image size variation
  - Pixel stats per class
  - Pipeline performance
  - Any anomalies found

---

## 🤝 Working style for Claude Code

While running this:

- Show me the plan before each phase. I'll say go before you execute.
- After each phase, print a short summary of what was done.
- If something looks off in the data (weird outliers, broken files), pause and tell me — don't silently skip.
- Prefer readable code over clever code. Comments explain *why*, not *what*.

---

## 🔜 Next (don't do yet — planning ahead)

After this pipeline is solid:

1. Switch the same images through a **MONAI** pipeline and compare.
2. Add **data augmentation** (rotation, flip, intensity jitter).
3. Train a baseline CNN on the standardized data.