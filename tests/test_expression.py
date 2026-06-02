from __future__ import annotations

from math import cos, radians, sin

from src.config import Thresholds
from src.detector.expression import RIGHT_EYE, ExpressionDecision, ExpressionScorer
from src.detector.face import FaceDetectionResult, FaceLandmarks


def test_expression_drops_closed_eyes_when_not_profile():
    detection = FaceDetectionResult(
        file_path="mock.jpg",
        used_vision=False,
        faces=[FaceLandmarks(bbox=(0.25, 0.2, 0.5, 0.5), confidence=1.0, points=_closed_eye_points())],
    )

    result = ExpressionScorer(Thresholds(eye_aspect_ratio=0.26)).score(detection)

    assert result.decision == ExpressionDecision.DROP
    assert result.eyes_closed
    assert result.reason == "闭眼/眯眼 (表情管理失败)"


def test_expression_drops_squint_when_not_profile():
    detection = FaceDetectionResult(
        file_path="mock.jpg",
        used_vision=False,
        faces=[FaceLandmarks(bbox=(0.25, 0.2, 0.5, 0.5), confidence=1.0, points=_squint_points())],
    )

    result = ExpressionScorer(Thresholds(eye_aspect_ratio=0.26)).score(detection)

    assert result.decision == ExpressionDecision.DROP
    assert result.eyes_closed
    assert result.eye_aspect_ratio is not None
    assert result.eye_aspect_ratio < 0.26


def test_expression_bypasses_closed_eye_drop_for_true_profile():
    points = _closed_eye_points()
    _move_eye(points, RIGHT_EYE, dx=-0.28)
    points[1] = (0.68, 0.52)
    detection = FaceDetectionResult(
        file_path="profile.jpg",
        used_vision=False,
        faces=[FaceLandmarks(bbox=(0.20, 0.2, 0.6, 0.5), confidence=1.0, points=points)],
    )

    result = ExpressionScorer(Thresholds(eye_aspect_ratio=0.26)).score(detection)

    assert ExpressionScorer._is_profile_face(detection.primary_face)
    assert result.decision == ExpressionDecision.PASS
    assert not result.eyes_closed
    assert "Profile face detected; EAR check bypassed." in result.notes


def test_profile_requires_nose_edge_and_small_frontal_ratio():
    profile_points = _open_eye_points()
    _move_eye(profile_points, RIGHT_EYE, dx=-0.28)
    profile_points[1] = (0.68, 0.52)
    assert ExpressionScorer._is_profile_face(
        FaceLandmarks(bbox=(0.20, 0.2, 0.6, 0.5), confidence=1.0, points=profile_points)
    )

    center_nose = profile_points.copy()
    center_nose[1] = (0.50, 0.52)
    assert not ExpressionScorer._is_profile_face(
        FaceLandmarks(bbox=(0.20, 0.2, 0.6, 0.5), confidence=1.0, points=center_nose)
    )

    wide_eyes = _open_eye_points()
    wide_eyes[1] = (0.68, 0.52)
    assert not ExpressionScorer._is_profile_face(
        FaceLandmarks(bbox=(0.20, 0.2, 0.6, 0.5), confidence=1.0, points=wide_eyes)
    )


def test_profile_can_fall_back_to_nose_edge_when_eye_points_missing():
    points = _open_eye_points()
    for index in RIGHT_EYE:
        points.pop(index)
    points[1] = (0.68, 0.52)

    assert ExpressionScorer._is_profile_face(
        FaceLandmarks(bbox=(0.20, 0.2, 0.6, 0.5), confidence=1.0, points=points)
    )


def test_expression_open_eyes_are_pass():
    detection = FaceDetectionResult(
        file_path="mock.jpg",
        used_vision=False,
        faces=[FaceLandmarks(bbox=(0.25, 0.2, 0.5, 0.5), confidence=1.0, points=_open_eye_points())],
    )

    result = ExpressionScorer().score(detection)

    assert result.decision == ExpressionDecision.PASS
    assert not result.eyes_closed
    assert result.score == 0.0


def test_expression_no_face_passes_to_downstream_aesthetic():
    detection = FaceDetectionResult(file_path="landscape.jpg", faces=[], used_vision=False)

    result = ExpressionScorer().score(detection)

    assert result.decision == ExpressionDecision.PASS
    assert result.reason is None
    assert result.eye_aspect_ratio is None


def test_expression_vision_face_without_landmarks_drops_occluded_portrait():
    detection = FaceDetectionResult(
        file_path="occluded.jpg",
        faces=[],
        used_vision=False,
        vision_face_count=1,
    )

    result = ExpressionScorer().score(detection)

    assert result.decision == ExpressionDecision.DROP
    assert result.score == -1.0
    assert result.reason == "人脸被严重遮挡或角度极差"


def test_expression_vision_face_with_missing_eye_landmarks_drops():
    detection = FaceDetectionResult(
        file_path="occluded.jpg",
        faces=[FaceLandmarks(bbox=(0.25, 0.2, 0.5, 0.5), confidence=1.0, points={})],
        used_vision=True,
        vision_face_count=1,
    )

    result = ExpressionScorer().score(detection)

    assert result.decision == ExpressionDecision.DROP
    assert result.reason == "人脸关键点不足或遮挡严重"


def test_expression_uses_configured_ear_threshold():
    detection = FaceDetectionResult(
        file_path="mock.jpg",
        used_vision=False,
        faces=[
            FaceLandmarks(
                bbox=(0.25, 0.2, 0.5, 0.5),
                confidence=1.0,
                points=_mild_squint_points(),
            )
        ],
    )

    strict = ExpressionScorer(Thresholds(eye_aspect_ratio=0.30)).score(detection)
    relaxed = ExpressionScorer(Thresholds(eye_aspect_ratio=0.26)).score(detection)

    assert strict.decision == ExpressionDecision.DROP
    assert relaxed.decision == ExpressionDecision.PASS
    assert strict.eye_aspect_ratio is not None
    assert 0.26 < strict.eye_aspect_ratio < 0.30


def test_eye_aspect_ratio_is_rotation_invariant():
    points = _open_eye_points()
    face = FaceLandmarks(bbox=(0.25, 0.2, 0.5, 0.5), confidence=1.0, points=points)
    baseline = ExpressionScorer._eye_aspect_ratio(face, (33, 160, 158, 133, 153, 144))

    rotated = points.copy()
    for index in (33, 160, 158, 133, 153, 144):
        rotated[index] = _rotate(points[index], origin=(0.40, 0.50), degrees=28)
    rotated_face = FaceLandmarks(
        bbox=(0.25, 0.2, 0.5, 0.5),
        confidence=1.0,
        points=rotated,
    )
    rotated_ear = ExpressionScorer._eye_aspect_ratio(rotated_face, (33, 160, 158, 133, 153, 144))

    assert baseline is not None
    assert rotated_ear is not None
    assert abs(rotated_ear - baseline) < 1e-9


def _closed_eye_points() -> dict[int, tuple[float, float]]:
    points = _open_eye_points()
    for idx in (160, 158, 385, 387):
        x, _ = points[idx]
        points[idx] = (x, 0.51)
    for idx in (153, 144, 373, 380):
        x, _ = points[idx]
        points[idx] = (x, 0.49)
    return points


def _squint_points() -> dict[int, tuple[float, float]]:
    points = _open_eye_points()
    for idx in (160, 158, 385, 387):
        x, _ = points[idx]
        points[idx] = (x, 0.535)
    for idx in (153, 144, 373, 380):
        x, _ = points[idx]
        points[idx] = (x, 0.465)
    return points


def _mild_squint_points() -> dict[int, tuple[float, float]]:
    points = _open_eye_points()
    for idx in (160, 158, 385, 387):
        x, _ = points[idx]
        points[idx] = (x, 0.5425)
    for idx in (153, 144, 373, 380):
        x, _ = points[idx]
        points[idx] = (x, 0.4575)
    return points


def _open_eye_points() -> dict[int, tuple[float, float]]:
    return {
        1: (0.50, 0.52),
        9: (0.50, 0.20),
        152: (0.50, 0.90),
        33: (0.25, 0.5),
        160: (0.35, 0.62),
        158: (0.45, 0.62),
        133: (0.55, 0.5),
        153: (0.45, 0.38),
        144: (0.35, 0.38),
        362: (0.60, 0.5),
        385: (0.70, 0.62),
        387: (0.80, 0.62),
        263: (0.90, 0.5),
        373: (0.80, 0.38),
        380: (0.70, 0.38),
        78: (0.40, 0.75),
        13: (0.50, 0.77),
        308: (0.60, 0.75),
        14: (0.50, 0.73),
    }


def _move_eye(points: dict[int, tuple[float, float]], indexes: tuple[int, ...], *, dx: float) -> None:
    for index in indexes:
        x, y = points[index]
        points[index] = (x + dx, y)


def _rotate(
    point: tuple[float, float],
    *,
    origin: tuple[float, float],
    degrees: float,
) -> tuple[float, float]:
    angle = radians(degrees)
    px, py = point[0] - origin[0], point[1] - origin[1]
    return (
        origin[0] + px * cos(angle) - py * sin(angle),
        origin[1] + px * sin(angle) + py * cos(angle),
    )
