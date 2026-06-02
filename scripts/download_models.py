"""Print model download guidance.

Real model hosting is project-specific, so this script creates the model directory
and tells the user exactly which artifacts the runtime expects.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_DIR  # noqa: E402
from src.models.model_registry import MODEL_REGISTRY  # noqa: E402


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {MODEL_DIR}")
    for spec in MODEL_REGISTRY:
        status = "present" if spec.exists else "missing"
        print(f"- {spec.filename}: {status} ({spec.task})")
    print("Place Core ML .mlpackage artifacts in this directory to enable ML scoring.")


if __name__ == "__main__":
    main()
