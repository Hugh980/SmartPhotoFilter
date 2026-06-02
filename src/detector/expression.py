"""Strict expression and face-state screening from Apple Vision landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot

from src.config import Thresholds
from src.detector.face import FaceDetectionResult, FaceLandmarks


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH = (78, 13, 308, 14)
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)
NOSE_TIP = 1


class ExpressionDecision(str, Enum):
    """Strict expression gate decision."""

    PASS = "pass"
    REVIEW = "review"
    DROP = "drop"


@dataclass(frozen=True)
class ExpressionResult:
    """Expression and face-state quality result."""

    decision: ExpressionDecision
    score: float
    eyes_closed: bool
    mouth_open: bool
    gaze_off_center: bool
    eye_aspect_ratio: float | None
    mouth_aspect_ratio: float | None
    reason: str | None = None
    left_eye_aspect_ratio: float | None = None
    right_eye_aspect_ratio: float | None = None
    notes: tuple[str, ...] = ()


class ExpressionScorer:
    """Screen expression state from eye, mouth, and gaze landmarks."""

    def __init__(self, thresholds: Thresholds = Thresholds()) -> None:
        self.thresholds = thresholds

    def score(self, detection: FaceDetectionResult) -> ExpressionResult:
        face = detection.primary_face
        if face is None:
            if detection.vision_face_count > 0:
                return ExpressionResult(
                    decision=ExpressionDecision.DROP,
                    score=-1.0,
                    eyes_closed=False,
                    mouth_open=False,
                    gaze_off_center=False,
                    eye_aspect_ratio=None,
                    mouth_aspect_ratio=None,
                    reason="人脸被严重遮挡或角度极差",
                    notes=(
                        "Vision detected face, but landmarks were unavailable. "
                        "Dropping as poor portrait.",
                    ),
                )
            return ExpressionResult(
                decision=ExpressionDecision.PASS,
                score=0.0,
                eyes_closed=False,
                mouth_open=False,
                gaze_off_center=False,
                eye_aspect_ratio=None,
                mouth_aspect_ratio=None,
                notes=("No face landmarks; expression passed through to aesthetic model.",),
            )

        left_ear = self._eye_aspect_ratio(face, LEFT_EYE)
        right_ear = self._eye_aspect_ratio(face, RIGHT_EYE)
        visible_ears = [ear for ear in (left_ear, right_ear) if ear is not None]
        ear_score = min(visible_ears) if visible_ears else None
        mar = self._mouth_aspect_ratio(face)
        gaze_off = self._gaze_off_center(face)
        is_profile = self._is_profile_face(face)
        notes: list[str] = []
        if is_profile:
            notes.append("Profile face detected; EAR check bypassed.")

        if not visible_ears:
            return ExpressionResult(
                decision=ExpressionDecision.DROP,
                score=-1.0,
                eyes_closed=False,
                mouth_open=False,
                gaze_off_center=gaze_off,
                eye_aspect_ratio=None,
                mouth_aspect_ratio=mar,
                reason="人脸关键点不足或遮挡严重",
                left_eye_aspect_ratio=left_ear,
                right_eye_aspect_ratio=right_ear,
                notes=(
                    *notes,
                    "Vision detected face, but eye landmarks were unavailable.",
                ),
            )

        if visible_ears and any(ear < self.thresholds.eye_aspect_ratio for ear in visible_ears):
            if not is_profile:
                return ExpressionResult(
                    decision=ExpressionDecision.DROP,
                    score=-1.0,
                    eyes_closed=True,
                    mouth_open=False,
                    gaze_off_center=gaze_off,
                    eye_aspect_ratio=ear_score,
                    mouth_aspect_ratio=mar,
                    reason="闭眼/眯眼 (表情管理失败)",
                    left_eye_aspect_ratio=left_ear,
                    right_eye_aspect_ratio=right_ear,
                    notes=("Strict EAR threshold triggered.",),
                )

        score = 0.0
        mouth_open = mar is not None and mar > 0.55
        if mouth_open:
            score -= 0.3
            notes.append("Open mouth detected.")
        if gaze_off:
            score -= 0.5
            notes.append("Gaze appears off-center.")

        decision = ExpressionDecision.REVIEW if score <= -0.5 else ExpressionDecision.PASS
        reason = "视线偏移或表情需复核" if decision == ExpressionDecision.REVIEW else None

        return ExpressionResult(
            decision=decision,
            score=score,
            eyes_closed=False,
            mouth_open=mouth_open,
            gaze_off_center=gaze_off,
            eye_aspect_ratio=ear_score,
            mouth_aspect_ratio=mar,
            reason=reason,
            left_eye_aspect_ratio=left_ear,
            right_eye_aspect_ratio=right_ear,
            notes=tuple(notes),
        )

    @staticmethod
    def _eye_aspect_ratio(face: FaceLandmarks, indexes: tuple[int, ...]) -> float | None:
        points = [face.points[idx] for idx in indexes if idx in face.points]
        if len(points) < 4:
            return None

        corner_pair = max(
            ((first, second) for idx, first in enumerate(points) for second in points[idx + 1 :]),
            key=lambda pair: _distance(pair[0], pair[1]),
        )
        first_corner, second_corner = corner_pair
        width = _distance(first_corner, second_corner)
        if width == 0:
            return None

        dx = second_corner[0] - first_corner[0]
        dy = second_corner[1] - first_corner[1]
        max_above = 0.0
        max_below = 0.0
        for point in points:
            if point in corner_pair:
                continue
            signed_distance = (
                dx * (point[1] - first_corner[1])
                - dy * (point[0] - first_corner[0])
            ) / width
            if signed_distance >= 0:
                max_above = max(max_above, signed_distance)
            else:
                max_below = max(max_below, abs(signed_distance))

        height = max_above + max_below
        return height / width

    @staticmethod
    def _is_profile_face(face: FaceLandmarks) -> bool:
        if NOSE_TIP not in face.points:
            return False

        nose = face.points[NOSE_TIP]
        bbox_x, _, bbox_width, _ = face.bbox
        relative_nose_x = (nose[0] - bbox_x) / max(bbox_width, 1e-6)
        is_nose_at_edge = relative_nose_x < 0.25 or relative_nose_x > 0.75

        left_pts = [face.points[idx] for idx in LEFT_EYE if idx in face.points]
        right_pts = [face.points[idx] for idx in RIGHT_EYE if idx in face.points]
        if not left_pts or not right_pts:
            return is_nose_at_edge

        left_center = _mean_point(left_pts)
        right_center = _mean_point(right_pts)
        eye_dist = _distance(left_center, right_center)
        frontal_ratio = eye_dist / max(bbox_width, 1e-6)
        return frontal_ratio < 0.24 and is_nose_at_edge

    @staticmethod
    def _mouth_aspect_ratio(face: FaceLandmarks) -> float | None:
        try:
            left, top, right, bottom = [face.points[idx] for idx in MOUTH]
        except KeyError:
            return None
        width = _distance(left, right)
        if width == 0:
            return None
        return _distance(top, bottom) / width

    @staticmethod
    def _gaze_off_center(face: FaceLandmarks) -> bool:
        left_eye = [face.points[idx] for idx in LEFT_EYE if idx in face.points]
        right_eye = [face.points[idx] for idx in RIGHT_EYE if idx in face.points]
        if not left_eye or not right_eye:
            return False

        left_iris = [face.points[idx] for idx in LEFT_IRIS if idx in face.points]
        right_iris = [face.points[idx] for idx in RIGHT_IRIS if idx in face.points]
        if not left_iris or not right_iris:
            return False

        left_center = _mean_point(left_eye)
        right_center = _mean_point(right_eye)
        iris_center = _mean_point(left_iris + right_iris)
        eye_center = (
            (left_center[0] + right_center[0]) / 2.0,
            (left_center[1] + right_center[1]) / 2.0,
        )
        eye_width = max(_distance(left_center, right_center), 1e-6)
        return abs(iris_center[0] - eye_center[0]) / eye_width > 0.18


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )
