"""
Phase 3 — End-to-end smoke test for the preprocessing pipeline.

Loads 32 random images via the NumPy/PIL path, measures per-image timing, then
runs one batch through the tf.data path for comparison. Prints PASS/FAIL.

Brief target: "more than a few seconds for 32 images is off" → we flag if the
NumPy path total exceeds 3.0 s on a CPU.
"""
from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import IMAGE_SIZE, build_tf_dataset, preprocess  # noqa: E402

TRAIN = ROOT / "data" / "raw" / "Training"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
N = 32
SLOW_THRESHOLD_S = 3.0  # total for 32 images
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def pick_files(n: int) -> list[Path]:
    pool = []
    per_class = n // len(CLASSES)
    for cls in CLASSES:
        pool.extend(random.sample(list((TRAIN / cls).glob("*.jpg")), per_class))
    while len(pool) < n:
        cls = random.choice(CLASSES)
        pool.append(random.choice(list((TRAIN / cls).glob("*.jpg"))))
    random.shuffle(pool)
    return pool[:n]


def banner(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def main() -> int:
    files = pick_files(N)

    banner(f"NumPy/PIL path — preprocess() x {N}")
    timings: list[float] = []
    out_shapes: set[tuple] = set()
    out_dtypes: set[str] = set()
    nans = 0
    for fp in files:
        t0 = time.perf_counter()
        arr = preprocess(fp)
        timings.append(time.perf_counter() - t0)
        out_shapes.add(arr.shape)
        out_dtypes.add(str(arr.dtype))
        if not np.isfinite(arr).all():
            nans += 1

    total = sum(timings)
    print(f"  total:  {total*1000:7.1f} ms ({total:.3f} s)")
    print(f"  mean:   {statistics.mean(timings)*1000:7.2f} ms")
    print(f"  median: {statistics.median(timings)*1000:7.2f} ms")
    print(f"  min:    {min(timings)*1000:7.2f} ms")
    print(f"  max:    {max(timings)*1000:7.2f} ms")
    print(f"  output shapes:  {out_shapes}")
    print(f"  output dtypes:  {out_dtypes}")
    print(f"  NaN/Inf images: {nans}")

    banner("tf.data path — one batch of 32 through build_tf_dataset()")
    ds = build_tf_dataset(TRAIN, batch_size=N, image_size=IMAGE_SIZE, shuffle=True)
    it = iter(ds)
    _ = next(it)  # warm-up (first batch includes graph build + file listing)
    t0 = time.perf_counter()
    x, y = next(it)
    tf_batch_s = time.perf_counter() - t0
    print(f"  batch time (post-warmup):  {tf_batch_s*1000:7.1f} ms")
    print(f"  batch shape:               {tuple(x.shape)}  dtype={x.dtype.name}")
    print(f"  labels in batch:           {sorted(set(int(v) for v in y.numpy().tolist()))}")

    banner("Smoke test summary")
    checks = [
        ("all shapes (224,224,3)", out_shapes == {(*IMAGE_SIZE, 3)}),
        ("all dtypes float32", out_dtypes == {"float32"}),
        ("no NaN/Inf", nans == 0),
        (f"NumPy total < {SLOW_THRESHOLD_S}s for {N} images", total < SLOW_THRESHOLD_S),
        ("tf.data batch shape (32,224,224,3) float32",
         tuple(x.shape) == (N, *IMAGE_SIZE, 3) and x.dtype.name == "float32"),
    ]
    all_pass = True
    for name, ok in checks:
        flag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{flag}] {name}")
    print()
    print("ALL CHECKS PASSED" if all_pass else "ONE OR MORE CHECKS FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
