"""Aesthetic scoring with Core ML-ready fallback heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.detector.face import FaceDetectionResult
from src.models.coreml_runtime import CoreMLRuntime
from src.models.model_registry import NIMA_MODEL
from src.utils.image_utils import clamp, load_rgb_image, luminance, normalized_entropy, safe_mean

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class AestheticResult:
    """Aesthetic scores on a 0..10 scale."""

    nima_score: float
    composition_score: float
    color_harmony_score: float
    portrait_score: float
    final_score: float
    used_model: bool = False


class AestheticScorer:
    """NIMA-compatible scorer with deterministic CV heuristics."""

    def __init__(self, model_path: str | Path | None = None, use_model: bool = True) -> None:
        model = Path(model_path) if model_path is not None else NIMA_MODEL.path
        self._runtime = CoreMLRuntime(model) if use_model else None

    def score(
        self,
        image_path: str | Path,
        face_detection: FaceDetectionResult | None = None,
    ) -> AestheticResult:
        image = load_rgb_image(image_path, max_side=1280)
        return self.score_array(image, face_detection=face_detection)

    def score_array(
        self,
        image: np.ndarray,
        face_detection: FaceDetectionResult | None = None,
    ) -> AestheticResult:
        model_score = self._model_score(image)
        used_model = model_score is not None

        nima = model_score if model_score is not None else self._fallback_nima(image)
        composition = self._composition_score(image)
        color = self._color_harmony_score(image)
        portrait = self._portrait_score(face_detection)
        final = 0.40 * nima + 0.25 * composition + 0.15 * color + 0.20 * portrait

        return AestheticResult(
            nima_score=round(nima, 3),
            composition_score=round(composition, 3),
            color_harmony_score=round(color, 3),
            portrait_score=round(portrait, 3),
            final_score=round(final, 3),
            used_model=used_model,
        )

    def _model_score(self, image: np.ndarray) -> float | None:
        if self._runtime is None or not self._runtime.available:
            return None
        try:
            from PIL import Image

            resized = Image.fromarray(image, mode="RGB").resize((224, 224))
            prediction = self._runtime.predict({"image": resized})
        except Exception:
            return None
        for key in ("nima_score", "aesthetic_score", "score"):
            if key in prediction:
                value = prediction[key]
                if isinstance(value, (list, tuple, np.ndarray)):
                    value = value[0]
                return clamp(float(value), 0.0, 10.0)
        return None

    @staticmethod
    def _fallback_nima(image: np.ndarray) -> float:
        lum = luminance(image)
        exposure = 1.0 - min(abs(safe_mean(lum) - 128.0) / 128.0, 1.0)
        contrast = clamp(float(np.std(lum)) / 64.0)
        entropy = normalized_entropy(lum)

        rgb = image.astype(np.float32) / 255.0
        saturation = (np.max(rgb, axis=2) - np.min(rgb, axis=2))
        sat_score = clamp(float(np.mean(saturation)) / 0.35)

        return 10.0 * (0.30 * exposure + 0.30 * contrast + 0.25 * entropy + 0.15 * sat_score)

    @staticmethod
    def _composition_score(image: np.ndarray) -> float:
        gray = luminance(image).astype(np.float32)
        if cv2 is not None:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            energy = np.sqrt(gx * gx + gy * gy)
        else:
            gx = np.gradient(gray, axis=1)
            gy = np.gradient(gray, axis=0)
            energy = np.sqrt(gx * gx + gy * gy)

        total = float(energy.sum())
        if total <= 0:
            return 4.5

        height, width = energy.shape
        ys, xs = np.indices(energy.shape)
        cx = float((xs * energy).sum() / total) / max(width - 1, 1)
        cy = float((ys * energy).sum() / total) / max(height - 1, 1)

        thirds = [1.0 / 3.0, 2.0 / 3.0]
        distance = min(abs(cx - tx) + abs(cy - ty) for tx in thirds for ty in thirds)
        thirds_score = clamp(1.0 - distance / 0.75)

        center_distance = abs(cx - 0.5) + abs(cy - 0.5)
        balance_score = clamp(1.0 - center_distance)
        edge_penalty = clamp(min(cx, 1 - cx, cy, 1 - cy) / 0.2)
        return 10.0 * (0.45 * thirds_score + 0.35 * balance_score + 0.20 * edge_penalty)

    @staticmethod
    def _color_harmony_score(image: np.ndarray) -> float:
        rgb = image.astype(np.float32)
        channel_means = rgb.reshape(-1, 3).mean(axis=0)
        spread = float(np.std(channel_means))
        balance = clamp(1.0 - spread / 80.0)
        lum_entropy = normalized_entropy(luminance(image))

        saturation = (np.max(rgb, axis=2) - np.min(rgb, axis=2)) / 255.0
        saturation_mean = float(np.mean(saturation))
        saturation_score = clamp(1.0 - abs(saturation_mean - 0.35) / 0.35)

        return 10.0 * (0.35 * balance + 0.35 * lum_entropy + 0.30 * saturation_score)

    @staticmethod
    def _portrait_score(face_detection: FaceDetectionResult | None) -> float:
        if face_detection is None or face_detection.face_count == 0:
            return 6.0
        if face_detection.face_count > 1:
            return 4.5

        face = face_detection.primary_face
        if face is None:
            return 6.0
        cx, cy = face.center
        size_score = clamp(1.0 - abs(face.area_ratio - 0.12) / 0.18)
        placement_score = clamp(1.0 - (abs(cx - 0.5) + abs(cy - 0.42)) / 0.8)
        return 10.0 * (0.55 * size_score + 0.45 * placement_score)
