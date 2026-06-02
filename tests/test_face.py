from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.detector.face import FaceDetectionResult, FaceDetector, FaceLandmarks


def test_face_detector_returns_apple_vision_results(monkeypatch):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    face = FaceLandmarks(
        bbox=(0.25, 0.2, 0.5, 0.5),
        confidence=0.9,
        points={33: (0.3, 0.4), 133: (0.5, 0.4)},
    )
    monkeypatch.setattr(FaceDetector, "_detect_vision_faces", staticmethod(lambda _image: [face]))

    result = FaceDetector().detect_array(image, file_path="faces.png")

    assert result.face_count == 1
    assert result.used_vision
    assert result.vision_face_count == 1
    assert result.primary_face == face


def test_face_detector_vision_error_fails_open(monkeypatch, capsys):
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    def fail(_image):
        raise RuntimeError("vision unavailable")

    monkeypatch.setattr(FaceDetector, "_detect_vision_faces", staticmethod(fail))

    result = FaceDetector().detect_array(image, file_path="broken.png")

    assert result == FaceDetectionResult(
        file_path="broken.png",
        faces=[],
        used_vision=False,
        vision_face_count=0,
    )
    debug = capsys.readouterr().out
    assert "[ERROR] Apple Vision face detection failed for broken.png: vision unavailable" in debug


def test_vision_landmarks_are_mapped_to_legacy_expression_indexes():
    observation = _Observation(
        bbox=_Rect(0.2, 0.2, 0.6, 0.6),
        landmarks=_Landmarks(
            left_eye=[
                (0.10, 0.50),
                (0.30, 0.70),
                (0.70, 0.70),
                (0.90, 0.50),
                (0.70, 0.30),
                (0.30, 0.30),
            ],
            right_eye=[
                (0.10, 0.50),
                (0.30, 0.70),
                (0.70, 0.70),
                (0.90, 0.50),
                (0.70, 0.30),
                (0.30, 0.30),
            ],
            nose=[(0.50, 0.40), (0.50, 0.20)],
            contour=[(0.50, 0.00), (0.50, 1.00)],
            all_points=[(0.50, 0.90), (0.50, 0.50), (0.50, 0.10)],
        ),
    )

    face = FaceDetector._face_from_vision_observation(observation)

    assert face is not None
    assert face.bbox == pytest.approx((0.2, 0.2, 0.6, 0.6))
    assert face.confidence == 0.9
    assert {33, 160, 158, 133, 153, 144}.issubset(face.points)
    assert {362, 385, 387, 263, 373, 380}.issubset(face.points)
    assert face.points[1] == (0.5, 0.68)
    assert face.points[9] == pytest.approx((0.5, 0.26))
    assert face.points[152] == pytest.approx((0.5, 0.8))


@dataclass(frozen=True)
class _Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def origin(self):
        return (self.x, self.y)

    @property
    def size(self):
        return (self.width, self.height)


class _Observation:
    def __init__(self, bbox: _Rect, landmarks: _Landmarks) -> None:
        self._bbox = bbox
        self._landmarks = landmarks

    def boundingBox(self):
        return self._bbox

    def landmarks(self):
        return self._landmarks

    def confidence(self):
        return 0.9


class _Landmarks:
    def __init__(
        self,
        left_eye: list[tuple[float, float]],
        right_eye: list[tuple[float, float]],
        nose: list[tuple[float, float]],
        contour: list[tuple[float, float]],
        all_points: list[tuple[float, float]],
    ) -> None:
        self._left_eye = _Region(left_eye)
        self._right_eye = _Region(right_eye)
        self._nose = _Region(nose)
        self._contour = _Region(contour)
        self._all_points = _Region(all_points)

    def leftEye(self):
        return self._left_eye

    def rightEye(self):
        return self._right_eye

    def nose(self):
        return self._nose

    def faceContour(self):
        return self._contour

    def allPoints(self):
        return self._all_points


class _Region:
    def __init__(self, points: list[tuple[float, float]]) -> None:
        self._points = points

    def normalizedPoints(self):
        return self._points

    def pointCount(self):
        return len(self._points)
