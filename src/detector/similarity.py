"""Apple Vision FeaturePrint helpers for near-duplicate photo detection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeaturePrint:
    """A Vision feature-print observation tied to its source path."""

    path: str
    observation: Any
    revision: int | None = None


def generate_feature_print(path: str | Path) -> FeaturePrint | None:
    """Generate a VNFeaturePrintObservation for one image, if Vision is available."""

    try:
        import Foundation
        import Vision
    except Exception:
        return None

    path = Path(path)
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    observation = _perform_feature_print_request(Vision, url, prefer_revision_2=True)
    if observation is None:
        observation = _perform_feature_print_request(Vision, url, prefer_revision_2=False)
    if observation is None:
        return None

    revision = getattr(observation, "requestRevision", None)
    revision_value = int(revision() if callable(revision) else revision) if revision else None
    return FeaturePrint(path=str(path), observation=observation, revision=revision_value)


def precompute_feature_prints(
    paths: list[str | Path],
    max_workers: int = 1,
) -> dict[str, FeaturePrint]:
    """Generate FeaturePrints concurrently so post-processing stays off the UI thread."""

    if not paths:
        return {}

    workers = max(1, min(max_workers, len(paths)))
    feature_prints: dict[str, FeaturePrint] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(generate_feature_print, path): str(path) for path in paths}
        for future in as_completed(future_map):
            feature_print = future.result()
            if feature_print is not None:
                feature_prints[feature_print.path] = feature_print
    return feature_prints


def feature_print_distance(first: FeaturePrint, second: FeaturePrint) -> float | None:
    """Return Apple's FeaturePrint distance, or None when the bridge call fails."""

    method = getattr(first.observation, "computeDistance_toFeaturePrintObservation_error_", None)
    if method is None:
        return None

    try:
        result = method(None, second.observation, None)
    except TypeError:
        try:
            result = method(second.observation, None)
        except Exception:
            return None
    except Exception:
        return None

    if isinstance(result, tuple):
        for value in result:
            if isinstance(value, bool):
                continue
            if isinstance(value, (float, int)):
                return float(value)
        return None
    if isinstance(result, (float, int)):
        return float(result)
    return None


def _perform_feature_print_request(
    Vision: Any,
    url: Any,
    *,
    prefer_revision_2: bool,
) -> Any | None:
    request = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
    if prefer_revision_2:
        _set_revision_2_if_available(Vision, request)

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
    ok, _error = handler.performRequests_error_([request], None)
    if not ok:
        return None

    results = list(request.results() or [])
    return results[0] if results else None


def _set_revision_2_if_available(Vision: Any, request: Any) -> None:
    revision = getattr(Vision, "VNGenerateImageFeaturePrintRequestRevision2", None)
    if revision is None or not hasattr(request, "setRevision_"):
        return
    try:
        request.setRevision_(int(revision))
    except Exception:
        pass
