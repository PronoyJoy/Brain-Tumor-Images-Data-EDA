# Phase 3 — Smoke Test

## Goal
Confirm the whole pipeline chains together end-to-end and runs within a reasonable budget on CPU.

## How to reproduce
```bash
python src/smoke_test.py
```

## Timings (32 images, CPU)

| Path | Total time | Per image (mean) | Per image (median) | Min | Max |
|---|---|---|---|---|---|
| NumPy / PIL `preprocess()` | **194.5 ms** | 6.08 ms | 5.38 ms | 2.33 ms | 39.26 ms |
| tf.data `build_tf_dataset` (post-warmup, 32-batch) | **25.2 ms** | 0.79 ms | — | — | — |

tf.data is ~8× faster per image thanks to parallel decoding and graph fusion. Both paths are well under the brief's "more than a few seconds for 32 images is off" threshold.

## Checks (5 / 5 PASS)
- All output shapes `(224, 224, 3)`.
- All dtypes `float32`.
- No NaN / Inf.
- NumPy total < 3 s for 32 images.
- tf.data batch shape `(32, 224, 224, 3)` float32.

## Files added
- `src/smoke_test.py`
