"""Image loading and numeric helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_rgb_image(path: str | Path, max_side: int | None = None) -> np.ndarray:
    """Load an image as an RGB uint8 numpy array.

    EXIF orientation is applied so downstream geometry matches what users see in
    Preview or Photos.
    """

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if max_side is not None:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def save_rgb_image(path: str | Path, array: np.ndarray) -> None:
    """Save an RGB uint8 array for fixtures and generated outputs."""

    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def resize_long_edge(image: np.ndarray, max_side: int) -> np.ndarray:
    """Resize an RGB array while preserving aspect ratio."""

    height, width = image.shape[:2]
    current = max(height, width)
    if current <= max_side:
        return image

    scale = max_side / current
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    pil = Image.fromarray(image, mode="RGB")
    return np.asarray(pil.resize(new_size, Image.Resampling.LANCZOS), dtype=np.uint8)


def luminance(image: np.ndarray) -> np.ndarray:
    """Return Rec. 709 luminance as float32 in [0, 255]."""

    rgb = image.astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def normalized_entropy(values: np.ndarray, bins: int = 64) -> float:
    """Return histogram entropy normalized to [0, 1]."""

    hist, _ = np.histogram(values.ravel(), bins=bins, range=(0, 255), density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    probabilities = hist[hist > 0] / total
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy / np.log2(bins))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float value to the given range."""

    return max(low, min(high, float(value)))


def safe_mean(values: np.ndarray) -> float:
    """Mean that returns 0 for empty arrays."""

    if values.size == 0:
        return 0.0
    return float(np.mean(values))
