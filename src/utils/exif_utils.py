"""EXIF metadata helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import ExifTags, Image


def read_exif(path: str | Path) -> dict[str, Any]:
    """Read a compact set of EXIF fields with Pillow.

    The function never raises for unsupported image formats; scanner and reports can
    still proceed when metadata is missing.
    """

    try:
        with Image.open(path) as image:
            raw = image.getexif()
            if not raw:
                return {}
            named: dict[str, Any] = {}
            for key, value in raw.items():
                label = ExifTags.TAGS.get(key, str(key))
                if isinstance(value, bytes):
                    continue
                named[label] = value
            return named
    except Exception:
        return {}


def camera_label(exif: dict[str, Any]) -> str:
    """Return a friendly camera string for reports."""

    make = str(exif.get("Make", "")).strip()
    model = str(exif.get("Model", "")).strip()
    return " ".join(part for part in [make, model] if part) or "Unknown camera"
