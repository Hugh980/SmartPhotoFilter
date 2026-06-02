"""Apple Vision face detection and landmark extraction."""

from __future__ import annotations

import io
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.utils.image_utils import load_rgb_image


LEFT_EYE_INDEXES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDEXES = (362, 385, 387, 263, 373, 380)
NOSE_TIP_INDEX = 1
GLABELLA_INDEX = 9
CHIN_INDEX = 152


@dataclass(frozen=True)
class FaceLandmarks:
    """A single face with normalized landmarks and bounding box."""

    bbox: tuple[float, float, float, float]
    confidence: float
    points: dict[int, tuple[float, float]]

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0

    @property
    def area_ratio(self) -> float:
        return max(0.0, self.bbox[2]) * max(0.0, self.bbox[3])


@dataclass(frozen=True)
class FaceDetectionResult:
    file_path: str
    faces: list[FaceLandmarks]
    used_vision: bool
    vision_face_count: int = 0

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def primary_face(self) -> FaceLandmarks | None:
        if not self.faces:
            return None
        return max(self.faces, key=lambda face: face.area_ratio)


class FaceDetector:
    """Apple Vision face detector with landmarks mapped to the legacy point layout."""

    def __init__(self, max_faces: int = 8) -> None:
        self.max_faces = max_faces

    def detect(self, image_path: str | Path) -> FaceDetectionResult:
        image = load_rgb_image(image_path, max_side=1280)
        return self.detect_array(image, file_path=str(image_path))

    def detect_array(self, image: np.ndarray, file_path: str = "<array>") -> FaceDetectionResult:
        try:
            faces = self._detect_vision_faces(image)[: self.max_faces]
        except Exception as e:
            print(f"[ERROR] Apple Vision face detection failed for {file_path}: {e}")
            traceback.print_exc()
            return FaceDetectionResult(
                file_path=file_path,
                faces=[],
                used_vision=False,
                vision_face_count=0,
            )

        return FaceDetectionResult(
            file_path=file_path,
            faces=faces,
            used_vision=True,
            vision_face_count=len(faces),
        )

    def close(self) -> None:
        """Apple Vision requests do not hold persistent resources."""

    @staticmethod
    def _detect_vision_face_count(image: np.ndarray) -> int:
        """Compatibility helper for tests and debug output."""

        try:
            return len(FaceDetector._detect_vision_faces(image))
        except Exception as e:
            print(f"[ERROR] Apple Vision face detection failed: {e}")
            traceback.print_exc()
            return 0

    @staticmethod
    def _detect_vision_faces(image: np.ndarray) -> list[FaceLandmarks]:
        """Run VNDetectFaceLandmarksRequest and map results into FaceLandmarks."""

        try:
            import Quartz
            import Vision
            from CoreFoundation import CFDataCreate, kCFAllocatorDefault
        except Exception as e:
            print(f"[ERROR] Apple Vision face detection failed: {e}")
            traceback.print_exc()
            return []

        buffer = io.BytesIO()
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
        data = buffer.getvalue()
        cf_data = CFDataCreate(kCFAllocatorDefault, data, len(data))
        image_source = Quartz.CGImageSourceCreateWithData(cf_data, None)
        if image_source is None:
            return []
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if cg_image is None:
            return []

        request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f"VNDetectFaceLandmarksRequest failed: {error}")

        observations = list(request.results() or [])
        faces = [FaceDetector._face_from_vision_observation(observation) for observation in observations]
        return [face for face in faces if face is not None]

    @staticmethod
    def _face_from_vision_observation(observation: Any) -> FaceLandmarks | None:
        bbox = _vision_rect_to_bbox(observation.boundingBox())
        landmarks = observation.landmarks()
        if landmarks is None:
            return FaceLandmarks(
                bbox=bbox,
                confidence=_vision_confidence(observation),
                points={},
            )

        points: dict[int, tuple[float, float]] = {}
        points.update(_map_eye_points(_region_points(landmarks.leftEye(), bbox), LEFT_EYE_INDEXES))
        points.update(_map_eye_points(_region_points(landmarks.rightEye(), bbox), RIGHT_EYE_INDEXES))

        nose_points = _region_points(landmarks.nose(), bbox)
        all_points = _region_points(landmarks.allPoints(), bbox)
        contour_points = _region_points(_landmark_region(landmarks, "faceContour"), bbox)

        nose_tip = _lowest_point(nose_points)
        if nose_tip is not None:
            points[NOSE_TIP_INDEX] = nose_tip

        glabella = _upper_midpoint(all_points, bbox)
        if glabella is not None:
            points[GLABELLA_INDEX] = glabella

        chin = _lowest_point(contour_points or all_points)
        if chin is not None:
            points[CHIN_INDEX] = chin

        return FaceLandmarks(
            bbox=bbox,
            confidence=_vision_confidence(observation),
            points=points,
        )


def _vision_rect_to_bbox(rect: Any) -> tuple[float, float, float, float]:
    origin = _rect_component(rect, "origin")
    size = _rect_component(rect, "size")
    x = _coord_value(origin, "x")
    y = _coord_value(origin, "y")
    width = _coord_value(size, "width")
    height = _coord_value(size, "height")
    return (
        _clamp01(x),
        _clamp01(1.0 - y - height),
        _clamp01(width),
        _clamp01(height),
    )


def _region_points(region: Any, bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    if region is None:
        return []
    points = _normalized_points(region)
    x, y, width, height = bbox
    return [
        (
            _clamp01(x + _point_value(point, "x") * width),
            _clamp01(y + (1.0 - _point_value(point, "y")) * height),
        )
        for point in points
    ]


def _normalized_points(region: Any) -> list[Any]:
    if hasattr(region, "normalizedPoints"):
        points = region.normalizedPoints()
        point_count = int(region.pointCount()) if hasattr(region, "pointCount") else len(points)
        return [points[idx] for idx in range(point_count)]
    if hasattr(region, "points"):
        return list(region.points())
    return list(region)


def _map_eye_points(
    points: list[tuple[float, float]],
    indexes: tuple[int, int, int, int, int, int],
) -> dict[int, tuple[float, float]]:
    if len(points) < 4:
        return {}

    left = min(points, key=lambda point: point[0])
    right = max(points, key=lambda point: point[0])
    center_y = sum(point[1] for point in points) / len(points)
    middle = [point for point in points if point not in (left, right)]
    upper = sorted(
        [point for point in middle if point[1] <= center_y] or middle,
        key=lambda point: point[0],
    )
    lower = sorted(
        [point for point in middle if point[1] > center_y] or middle,
        key=lambda point: point[0],
    )
    upper_left = upper[0]
    upper_right = upper[-1]
    lower_left = lower[0]
    lower_right = lower[-1]
    p1, p2, p3, p4, p5, p6 = indexes
    return {
        p1: left,
        p2: upper_left,
        p3: upper_right,
        p4: right,
        p5: lower_right,
        p6: lower_left,
    }


def _landmark_region(landmarks: Any, name: str) -> Any:
    attr = getattr(landmarks, name, None)
    if attr is None:
        return None
    return attr() if callable(attr) else attr


def _upper_midpoint(
    points: list[tuple[float, float]],
    bbox: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    if points:
        upper_points = sorted(points, key=lambda point: point[1])[: max(1, len(points) // 8)]
        return _mean_point(upper_points)
    x, y, width, height = bbox
    return x + width / 2.0, y + height * 0.18


def _lowest_point(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return max(points, key=lambda point: point[1])


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _vision_confidence(observation: Any) -> float:
    confidence = getattr(observation, "confidence", None)
    if confidence is None:
        return 1.0
    return float(confidence() if callable(confidence) else confidence)


def _rect_component(rect: Any, name: str) -> Any:
    attr = getattr(rect, name, None)
    if attr is not None:
        return attr() if callable(attr) else attr
    index = 0 if name == "origin" else 1
    return rect[index]


def _coord_value(value: Any, name: str) -> float:
    attr = getattr(value, name, None)
    if attr is not None:
        return float(attr() if callable(attr) else attr)
    index = 0 if name in {"x", "width"} else 1
    return float(value[index])


def _point_value(point: Any, name: str) -> float:
    attr = getattr(point, name, None)
    if attr is not None:
        return float(attr() if callable(attr) else attr)
    index = 0 if name == "x" else 1
    return float(point[index])


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
