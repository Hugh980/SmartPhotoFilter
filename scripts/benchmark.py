"""Synthetic benchmark for the fallback scoring pipeline."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RuntimeConfig  # noqa: E402
from src.pipeline.scanner import scan_inputs  # noqa: E402
from src.pipeline.scorer import process_batch  # noqa: E402
from src.utils.image_utils import save_rgb_image  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for index in range(12):
            image = _synthetic_image(index)
            save_rgb_image(folder / f"sample-{index:02d}.png", image)

        assets = scan_inputs([folder])
        started = time.perf_counter()
        results = process_batch(
            assets,
            RuntimeConfig(max_workers=4, use_optional_ml=False),
        )
        elapsed = time.perf_counter() - started
        print(f"Processed {len(results)} images in {elapsed:.3f}s")
        print(f"{elapsed / max(1, len(results)):.4f}s/image")


def _synthetic_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(640, 960, 3), dtype=np.uint8)
    if seed % 3 == 0:
        base[:, ::16, :] = 255
        base[::16, :, :] = 0
    return base


if __name__ == "__main__":
    main()
