"""
Phase 2.5 — Verify the preprocessing pipeline.

Checks:
  - preprocess() output shape, dtype, no NaN/Inf
  - per-image mean ~= 0 and std ~= 1 over a tf.data batch
  - 4 before/after side-by-side images saved as a figure

Prints a PASS/FAIL summary at the end.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import IMAGE_SIZE, build_tf_dataset, load_image, preprocess  # noqa: E402

RAW = ROOT / "data" / "raw"
TRAIN = RAW / "Training"
FIG_DIR = ROOT / "outputs" / "figures"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

results: list[tuple[str, bool, str]] = []  # (check name, passed, detail)


def record(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}] {name}{(' — ' + detail) if detail else ''}")


def banner(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def check_numpy_path() -> None:
    banner("Check 1 — NumPy/PIL preprocess() one image per class")
    for cls in CLASSES:
        fp = random.choice(list((TRAIN / cls).glob("*.jpg")))
        arr = preprocess(fp)
        record(
            f"{cls}: shape/dtype",
            arr.shape == (*IMAGE_SIZE, 3) and arr.dtype == np.float32,
            f"got shape={arr.shape}, dtype={arr.dtype}",
        )
        record(
            f"{cls}: finite",
            np.isfinite(arr).all(),
            f"NaN={np.isnan(arr).any()}, Inf={np.isinf(arr).any()}",
        )
        record(
            f"{cls}: mean~0, std~1",
            abs(arr.mean()) < 1e-3 and abs(arr.std() - 1.0) < 1e-3,
            f"mean={arr.mean():+.4f}, std={arr.std():.4f}",
        )


def check_tf_batch() -> None:
    banner("Check 2 — tf.data batch standardization")
    ds = build_tf_dataset(TRAIN, batch_size=32, image_size=IMAGE_SIZE, shuffle=True)
    x, y = next(iter(ds))
    x_np = x.numpy()

    record(
        "batch shape (B,H,W,3) float32",
        x_np.shape == (32, *IMAGE_SIZE, 3) and x_np.dtype == np.float32,
        f"shape={x_np.shape}, dtype={x_np.dtype}",
    )
    record("no NaN/Inf in batch", np.isfinite(x_np).all())

    per_image_means = x_np.mean(axis=(1, 2, 3))
    per_image_stds = x_np.std(axis=(1, 2, 3))
    record(
        "per-image mean ~ 0 across batch",
        np.all(np.abs(per_image_means) < 1e-3),
        f"max |mean| = {np.abs(per_image_means).max():.2e}",
    )
    record(
        "per-image std ~ 1 across batch",
        np.all(np.abs(per_image_stds - 1.0) < 1e-3),
        f"max |std-1| = {np.abs(per_image_stds - 1.0).max():.2e}",
    )

    print(f"  Labels in batch: {sorted(set(int(v) for v in y.numpy().tolist()))}")


def before_after_figure() -> None:
    banner("Check 3 — Before/after visual comparison (4 images)")
    fig, axes = plt.subplots(4, 2, figsize=(7, 12))
    samples = []
    for cls in CLASSES:
        fp = random.choice(list((TRAIN / cls).glob("*.jpg")))
        samples.append((cls, fp))

    for row, (cls, fp) in enumerate(samples):
        raw = load_image(fp)
        proc = preprocess(fp)
        # rescale standardized image to [0,1] for visualization only
        vis = (proc - proc.min()) / (proc.max() - proc.min() + 1e-8)

        axes[row, 0].imshow(raw)
        axes[row, 0].set_title(f"{cls} — raw {raw.shape[1]}x{raw.shape[0]}", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(vis)
        axes[row, 1].set_title(
            f"{cls} — preprocessed {IMAGE_SIZE[1]}x{IMAGE_SIZE[0]}  "
            f"(mean={proc.mean():+.2f}, std={proc.std():.2f})",
            fontsize=9,
        )
        axes[row, 1].axis("off")
    fig.tight_layout()
    out = FIG_DIR / "preprocessing_before_after.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    record("figure written", out.exists(), str(out.relative_to(ROOT)))


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    check_numpy_path()
    check_tf_batch()
    before_after_figure()

    banner("Verification summary")
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [name for name, ok, _ in results if not ok]
    print(f"{passed}/{total} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
