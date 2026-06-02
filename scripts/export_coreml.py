"""Core ML export entry point placeholder.

The training pipeline depends on the chosen PyTorch checkpoints. This script defines
the command surface and fails clearly until those checkpoints are supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoints to Core ML")
    parser.add_argument("--blur-checkpoint", type=Path)
    parser.add_argument("--nima-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.blur_checkpoint and not args.nima_checkpoint:
        raise SystemExit(
            "Provide --blur-checkpoint and/or --nima-checkpoint. "
            "The runtime already works without models through fallback CV scoring."
        )
    raise SystemExit("Checkpoint conversion is not implemented for arbitrary checkpoints yet.")


if __name__ == "__main__":
    main()
