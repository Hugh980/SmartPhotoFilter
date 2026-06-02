"""macOS and runtime capability helpers."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Capabilities shown in CLI and GUI status areas."""

    system: str
    machine: str
    cpu_count: int
    coreml_available: bool
    vision_available: bool

    @property
    def acceleration_label(self) -> str:
        if self.coreml_available and self.system == "Darwin":
            return "Core ML ready"
        if self.vision_available:
            return "Apple Vision ready"
        return "CPU fallback"


def detect_runtime_capabilities() -> RuntimeCapabilities:
    """Probe optional acceleration packages without importing heavy model code."""

    return RuntimeCapabilities(
        system=platform.system(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        coreml_available=_module_available("coremltools"),
        vision_available=_module_available("Vision"),
    )


def recommended_workers(limit: int | None = None) -> int:
    """Return a conservative worker count for image batch processing."""

    count = max(1, os.cpu_count() or 1)
    workers = max(1, count - 1)
    if limit is not None:
        workers = min(workers, max(1, limit))
    return workers


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False
