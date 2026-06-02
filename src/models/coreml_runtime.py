"""Small Core ML wrapper with graceful fallback behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CoreMLRuntime:
    """Lazy Core ML model loader.

    This class keeps model usage optional so the app can run its deterministic CV
    fallback path during development, tests, and on machines without coremltools.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        return self.model_path.exists() and self._load() is not None

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model = self._load()
        if model is None:
            raise RuntimeError(f"Core ML model unavailable: {self.model_path}")
        return dict(model.predict(inputs))

    def _load(self) -> Any | None:
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            return None
        try:
            import coremltools as ct

            self._model = ct.models.MLModel(str(self.model_path))
        except Exception:
            self._model = None
        return self._model
