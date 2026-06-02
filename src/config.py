"""Application configuration and threshold defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".webp",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class Thresholds:
    """Runtime scoring thresholds.

    Values mirror the project description while keeping the fallback CV path usable
    when Core ML models are not installed.
    """

    sharpness_variance: float = 100.0
    deep_blur_score: float = 0.4
    eye_aspect_ratio: float = 0.2
    expression_discard_score: float = -0.5
    aesthetic_keep_score: float = 6.0
    aesthetic_review_score: float = 4.5
    similarity_distance: float = 0.35


@dataclass(frozen=True)
class RuntimeConfig:
    """Batch processing configuration."""

    thresholds: Thresholds = Thresholds()
    max_workers: int | None = None
    copy_files: bool = True
    use_optional_ml: bool = True


DEFAULT_CONFIG = RuntimeConfig()
