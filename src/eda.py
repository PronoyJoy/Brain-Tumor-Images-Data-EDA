"""
Phase 1 — Exploratory Data Analysis for the Brain Tumor MRI dataset.

Run: python src/eda.py
"""
from __future__ import annotations

import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

SEED = 42
SAMPLES_PER_CLASS_DIMS = 200
SAMPLES_PER_CLASS_PIXELS = 150  # heavier work; smaller sample

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FIG_DIR = ROOT / "outputs" / "figures"
PROC_DIR = ROOT / "outputs" / "processed"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SPLITS = ["Training", "Testing"]

random.seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid")


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def list_files(split: str, cls: str) -> list[Path]:
    return sorted((RAW / split / cls).glob("*"))


# ---------- 1.1 Class distribution ----------
def class_distribution() -> pd.DataFrame:
    banner("1.1 Class distribution")
    rows = []
    for split in SPLITS:
        for cls in CLASSES:
            rows.append({"split": split, "class": cls, "count": len(list_files(split, cls))})
    df = pd.DataFrame(rows)
    print(df.pivot(index="class", columns="split", values="count").to_string())

    train = df[df.split == "Training"]["count"].values
    test = df[df.split == "Testing"]["count"].values
    print(f"\nTraining range: {train.min()}-{train.max()}  (imbalance ratio: {train.max()/train.min():.2f})")
    print(f"Testing  range: {test.min()}-{test.max()}  (imbalance ratio: {test.max()/test.min():.2f})")
    balanced = (train.max() == train.min()) and (test.max() == test.min())
    print(f"Balanced: {balanced}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.barplot(data=df, x="class", y="count", hue="split", ax=ax)
    ax.set_title("Class distribution — Training vs Testing")
    ax.set_ylabel("Image count")
    for container in ax.containers:
        ax.bar_label(container, fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "class_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)}")
    return df


# ---------- 1.2 Image dimension audit ----------
def dimension_audit() -> pd.DataFrame:
    banner(f"1.2 Image dimension audit ({SAMPLES_PER_CLASS_DIMS} samples per class, both splits)")
    rows = []
    for split in SPLITS:
        for cls in CLASSES:
            files = list_files(split, cls)
            sample = random.sample(files, min(SAMPLES_PER_CLASS_DIMS, len(files)))
            for fp in tqdm(sample, desc=f"{split}/{cls}", leave=False):
                try:
                    with Image.open(fp) as im:
                        w, h = im.size
                        ch = len(im.getbands())
                except (UnidentifiedImageError, OSError):
                    continue
                rows.append({
                    "split": split, "class": cls, "file": fp.name,
                    "width": w, "height": h, "channels": ch,
                    "file_size_kb": fp.stat().st_size / 1024,
                })
    df = pd.DataFrame(rows)
    out_csv = PROC_DIR / "dim_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv.relative_to(ROOT)}  ({len(df)} rows)")

    print("\nWidth/height/file_size summary:")
    print(df[["width", "height", "channels", "file_size_kb"]].describe().round(1).to_string())

    print("\nChannel counts:")
    print(df["channels"].value_counts().to_string())

    print("\nUnique (w,h) shapes — top 5:")
    shapes = df.groupby(["width", "height"]).size().sort_values(ascending=False).head(5)
    print(shapes.to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=df, x="width", y="height", hue="class", alpha=0.55, s=25, ax=ax)
    ax.set_title(f"Image dimensions — {len(df)} samples")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    fig.tight_layout()
    out = FIG_DIR / "dimension_scatter.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)}")
    return df


# ---------- 1.3 Pixel intensity statistics ----------
def pixel_intensities() -> pd.DataFrame:
    banner(f"1.3 Pixel intensity stats ({SAMPLES_PER_CLASS_PIXELS} samples per class, Training only)")
    rows = []
    histograms: dict[str, np.ndarray] = {c: np.zeros(256, dtype=np.int64) for c in CLASSES}
    for cls in CLASSES:
        files = list_files("Training", cls)
        sample = random.sample(files, min(SAMPLES_PER_CLASS_PIXELS, len(files)))
        for fp in tqdm(sample, desc=f"Training/{cls}", leave=False):
            try:
                with Image.open(fp) as im:
                    arr = np.asarray(im.convert("L"), dtype=np.uint8)
            except (UnidentifiedImageError, OSError):
                continue
            rows.append({
                "class": cls,
                "file": fp.name,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            })
            histograms[cls] += np.bincount(arr.ravel(), minlength=256)

    df = pd.DataFrame(rows)
    print("\nPer-class pixel statistics (grayscale 0-255):")
    summary = df.groupby("class").agg(
        mean_of_means=("mean", "mean"),
        std_of_means=("mean", "std"),
        mean_of_stds=("std", "mean"),
        n=("file", "count"),
    ).round(2)
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    for cls in CLASSES:
        h = histograms[cls].astype(float)
        h /= h.sum()  # density
        ax.plot(np.arange(256), h, label=cls, linewidth=1.4)
    ax.set_title("Pixel intensity distribution per class (grayscale, normalized)")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "pixel_intensity_per_class.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)}")
    return df


# ---------- 1.4 Sample 4x4 grid ----------
def sample_grid() -> None:
    banner("1.4 Sample 4x4 grid (4 random per class, Training)")
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    for row, cls in enumerate(CLASSES):
        files = list_files("Training", cls)
        sample = random.sample(files, 4)
        for col, fp in enumerate(sample):
            with Image.open(fp) as im:
                ax = axes[row, col]
                ax.imshow(im, cmap="gray")
                ax.set_title(f"{cls}\n{im.size[0]}x{im.size[1]}", fontsize=9)
                ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "sample_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)}")


# ---------- 1.5 Sanity checks ----------
def sanity_checks() -> dict:
    banner("1.5 Sanity checks")
    corrupted: list[Path] = []
    non_image: list[Path] = []
    filenames_by_class: dict[tuple[str, str], list[str]] = defaultdict(list)
    hash_to_files: dict[str, list[str]] = defaultdict(list)

    all_files = [(split, cls, fp) for split in SPLITS for cls in CLASSES for fp in list_files(split, cls)]
    for split, cls, fp in tqdm(all_files, desc="Scanning all files"):
        if fp.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            non_image.append(fp)
            continue
        try:
            with Image.open(fp) as im:
                im.verify()
        except Exception:
            corrupted.append(fp)
            continue
        filenames_by_class[(split, cls)].append(fp.name)
        # cheap content hash for duplicate detection (size + first/last 4KB)
        try:
            with open(fp, "rb") as f:
                head = f.read(4096)
                f.seek(max(0, fp.stat().st_size - 4096))
                tail = f.read(4096)
            h = hashlib.md5(head + tail + str(fp.stat().st_size).encode()).hexdigest()
            hash_to_files[h].append(f"{split}/{cls}/{fp.name}")
        except OSError:
            pass

    # duplicate filenames across classes within the same split
    cross_dups: list[tuple[str, str]] = []
    for split in SPLITS:
        seen: dict[str, str] = {}
        for cls in CLASSES:
            for name in filenames_by_class.get((split, cls), []):
                if name in seen and seen[name] != cls:
                    cross_dups.append((split, name))
                seen.setdefault(name, cls)

    # content duplicates (any pair of files with the same head+tail+size hash)
    content_dups = {h: lst for h, lst in hash_to_files.items() if len(lst) > 1}

    print(f"\nCorrupted images:       {len(corrupted)}")
    if corrupted[:5]:
        for fp in corrupted[:5]:
            print(f"  {fp}")
    print(f"Non-image files:        {len(non_image)}")
    print(f"Cross-class dup names:  {len(cross_dups)}")
    if cross_dups[:5]:
        for split, name in cross_dups[:5]:
            print(f"  {split}: {name}")
    print(f"Content-hash dup groups: {len(content_dups)}")
    if content_dups:
        for i, (h, lst) in enumerate(list(content_dups.items())[:5]):
            print(f"  [{i+1}] {len(lst)} files: {lst[:3]}{' ...' if len(lst) > 3 else ''}")

    return {
        "corrupted": len(corrupted),
        "non_image": len(non_image),
        "cross_class_filename_dups": len(cross_dups),
        "content_dup_groups": len(content_dups),
    }


def main() -> int:
    if not RAW.exists():
        print(f"ERROR: {RAW} not found.")
        return 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    dist = class_distribution()
    dims = dimension_audit()
    pixels = pixel_intensities()
    sample_grid()
    checks = sanity_checks()

    banner("Phase 1 — Final summary")
    train_counts = dist[dist.split == "Training"]["count"].values
    test_counts = dist[dist.split == "Testing"]["count"].values
    print(f"Classes: {CLASSES}")
    print(f"Training counts: {dict(zip(CLASSES, train_counts))}")
    print(f"Testing  counts: {dict(zip(CLASSES, test_counts))}")
    print(f"Balanced: train={train_counts.max() == train_counts.min()}  test={test_counts.max() == test_counts.min()}")
    print(f"\nImage dimensions (sampled): "
          f"w {int(dims.width.min())}-{int(dims.width.max())} (mean {dims.width.mean():.0f}), "
          f"h {int(dims.height.min())}-{int(dims.height.max())} (mean {dims.height.mean():.0f})")
    print(f"Unique channel counts: {sorted(dims.channels.unique().tolist())}")
    print(f"\nPer-class grayscale mean (of per-image means):")
    print(pixels.groupby('class')['mean'].mean().round(2).to_string())
    print(f"\nSanity: {checks}")
    print("\nFigures written to outputs/figures/:")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  - {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
