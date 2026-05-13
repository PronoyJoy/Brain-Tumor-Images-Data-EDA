"""
Phase 2 — Image preprocessing pipeline for the Brain Tumor MRI dataset.

Public API:
    load_image(path)               -> np.ndarray (H, W, 3) uint8
    resize_image(arr, size)        -> np.ndarray (size_h, size_w, 3) uint8
    samplewise_standardize(arr)    -> np.ndarray float32 (zero-mean, unit-variance per image)
    preprocess(path, size)         -> np.ndarray (size_h, size_w, 3) float32
    build_tf_dataset(directory, batch_size, image_size) -> tf.data.Dataset
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

IMAGE_SIZE: Tuple[int, int] = (224, 224)
EPS = 1e-8

# Dataset note: source images are a mix of 1-channel grayscale and 3-channel JPGs
# (EDA found ~44% single-channel in a 1600-image sample). We coerce everything to
# RGB so the same tensors can feed pretrained 3-channel models downstream.


def load_image(path: str | Path) -> np.ndarray:
    """Load an image and return an (H, W, 3) uint8 NumPy array."""
    with Image.open(path) as im:
        # convert("RGB") replicates grayscale into 3 channels and drops alpha
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def resize_image(arr: np.ndarray, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize to (height, width) with bilinear interpolation.

    Aspect ratio is NOT preserved — simple resize per the project brief. Trade-off:
    fast and trivially batchable, at the cost of distorting non-square inputs.
    Most images are 512x512, so the impact is limited but non-zero.
    """
    target_h, target_w = size
    im = Image.fromarray(arr)
    im = im.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def samplewise_standardize(arr: np.ndarray) -> np.ndarray:
    """Per-image standardization: (x - mean(x)) / (std(x) + eps).

    Returns float32. Stats are computed across the entire image (all channels +
    all pixels) so the channel-wise relationship is preserved.
    """
    a = arr.astype(np.float32)
    mean = a.mean()
    std = a.std()
    return (a - mean) / (std + EPS)


def preprocess(path: str | Path, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """End-to-end: load -> resize -> samplewise standardize. Returns float32."""
    return samplewise_standardize(resize_image(load_image(path), size))


def build_tf_dataset(
    directory: str | Path,
    batch_size: int = 32,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    shuffle: bool = True,
    seed: int = 42,
):
    """Build a tf.data.Dataset from a class-subfolder directory.

    `directory` should contain one subfolder per class (e.g. data/raw/Training/).
    Output element spec: ((batch, H, W, 3) float32, (batch,) int32) one-hot=False.
    Standardization is applied per-image inside the map.
    """
    import tensorflow as tf  # local import so non-TF callers stay light

    ds = tf.keras.utils.image_dataset_from_directory(
        str(directory),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",          # forces grayscale inputs to be replicated to 3ch
        batch_size=batch_size,
        image_size=image_size,     # bilinear resize is the default
        shuffle=shuffle,
        seed=seed,
    )

    def _standardize(x, y):
        x = tf.cast(x, tf.float32)
        # per-image stats across H, W, C  (axis=[1,2,3])
        mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        return (x - mean) / (std + EPS), y

    return ds.map(_standardize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
