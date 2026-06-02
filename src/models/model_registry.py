"""Model registry and expected artifact names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import MODEL_DIR


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    task: str
    required: bool = False

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.filename

    @property
    def exists(self) -> bool:
        return self.path.exists()


BLUR_MODEL = ModelSpec(
    name="EfficientNet-B0 blur detector",
    filename="blur_detector.mlpackage",
    task="deep blur scoring",
)

NIMA_MODEL = ModelSpec(
    name="MobileNetV3 NIMA aesthetic scorer",
    filename="nima_aesthetic.mlpackage",
    task="aesthetic scoring",
)

MODEL_REGISTRY = [BLUR_MODEL, NIMA_MODEL]


def model_status() -> list[dict[str, str | bool]]:
    """Return status dictionaries for CLI/reporting."""

    return [
        {
            "name": spec.name,
            "task": spec.task,
            "path": str(spec.path),
            "exists": spec.exists,
            "required": spec.required,
        }
        for spec in MODEL_REGISTRY
    ]
