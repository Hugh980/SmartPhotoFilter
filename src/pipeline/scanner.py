"""Filesystem scanner for image batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.config import SUPPORTED_IMAGE_EXTENSIONS
from src.utils.exif_utils import read_exif


@dataclass(frozen=True)
class ImageAsset:
    """A discovered image and lightweight metadata."""

    path: Path
    width: int | None = None
    height: int | None = None
    exif: dict | None = None

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()


def scan_inputs(inputs: Iterable[str | Path]) -> list[ImageAsset]:
    """Expand files and directories into supported images."""

    seen: set[Path] = set()
    assets: list[ImageAsset] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        candidates = _iter_candidates(path)
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            assets.append(_read_asset(candidate))
    return sorted(assets, key=lambda asset: str(asset.path))


def _iter_candidates(path: Path) -> Iterable[Path]:
    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                yield candidate.resolve()
    elif path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        yield path.resolve()


def _read_asset(path: Path) -> ImageAsset:
    width: int | None = None
    height: int | None = None
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        pass
    return ImageAsset(path=path, width=width, height=height, exif=read_exif(path))
