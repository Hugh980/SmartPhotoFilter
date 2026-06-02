"""Sharpness and blur detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import Thresholds
from src.models.coreml_runtime import CoreMLRuntime
from src.models.model_registry import BLUR_MODEL
from src.utils.image_utils import clamp, load_rgb_image, luminance

try:
    import cv2
except Exception:  # pragma: no cover - exercised only when OpenCV is absent
    cv2 = None


@dataclass(frozen=True)
class SharpnessResult:
    """Result of the two-stage blur detector."""

    file_path: str
    laplacian_variance: float
    blur_score: float
    is_blurry: bool
    threshold: float
    used_deep_model: bool = False
    rescued_by_roi: bool = False


class SharpnessDetector:
    """Rule-based Laplacian blur detector with optional Core ML refinement."""

    def __init__(
        self,
        thresholds: Thresholds = Thresholds(),
        model_path: str | Path | None = None,
        use_deep_model: bool = True,
    ) -> None:
        self.thresholds = thresholds
        model = Path(model_path) if model_path is not None else BLUR_MODEL.path
        self._runtime = CoreMLRuntime(model) if use_deep_model else None

    def analyze(self, image_path: str | Path) -> SharpnessResult:
        image = load_rgb_image(image_path, max_side=1600)
        return self.analyze_array(image, file_path=str(image_path))

    def analyze_array(self, image: np.ndarray, file_path: str = "<array>") -> SharpnessResult:
        variance = self.laplacian_variance(image)
        is_blurry = variance < self.thresholds.sharpness_variance
        rescued_by_roi = False

        if is_blurry and variance > 0:
            roi = self._get_roi(image)
            roi_variance = self.laplacian_variance(roi)
            if roi_variance >= self.thresholds.sharpness_variance:
                variance = roi_variance
                is_blurry = False
                rescued_by_roi = True

        fallback_score = self._variance_to_score(variance)
        deep_score = self._deep_blur_score(image)
        used_deep = deep_score is not None
        score = deep_score if deep_score is not None else fallback_score
        if used_deep and not rescued_by_roi:
            is_blurry = score < self.thresholds.deep_blur_score

        return SharpnessResult(
            file_path=file_path,
            laplacian_variance=variance,
            blur_score=score,
            is_blurry=is_blurry,
            threshold=self.thresholds.sharpness_variance,
            used_deep_model=used_deep,
            rescued_by_roi=rescued_by_roi,
        )

    def _get_roi(self, image: np.ndarray) -> np.ndarray:
        """Return the primary sharpness region: largest face or center crop.

        Bokeh portraits can have intentionally soft backgrounds, so global Laplacian
        variance may understate sharpness. This method focuses the rescue check on
        a face when available, then falls back to the central 50% crop.
        """

        if cv2 is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                classifier = cv2.CascadeClassifier(cascade_path)
                if not classifier.empty():
                    faces = classifier.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(24, 24),
                    )
                    if len(faces) > 0:
                        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                        return self._expanded_crop(image, int(x), int(y), int(w), int(h), 0.20)
            except Exception:
                pass

        return self._center_crop(image, fraction=0.50)

    @staticmethod
    def laplacian_variance(image: np.ndarray) -> float:
        """Compute variance of the Laplacian on luminance."""

        if cv2 is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

        gray = luminance(image)
        dxx = np.diff(gray, n=2, axis=1)
        dyy = np.diff(gray, n=2, axis=0)
        h = min(dxx.shape[0], dyy.shape[0])
        w = min(dxx.shape[1], dyy.shape[1])
        if h <= 0 or w <= 0:
            return 0.0
        laplace = dxx[:h, :w] + dyy[:h, :w]
        return float(np.var(laplace))

    def _deep_blur_score(self, image: np.ndarray) -> float | None:
        if self._runtime is None or not self._runtime.available:
            return None

        try:
            from PIL import Image

            resized = Image.fromarray(image, mode="RGB").resize((224, 224))
            prediction = self._runtime.predict({"image": resized})
        except Exception:
            return None

        for key in ("blur_score", "sharpness", "score"):
            if key in prediction:
                value = prediction[key]
                if isinstance(value, (list, tuple, np.ndarray)):
                    value = value[0]
                return clamp(float(value))
        return None

    @staticmethod
    def _expanded_crop(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        expansion: float,
    ) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        pad_x = int(width * expansion)
        pad_y = int(height * expansion)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(image_width, x + width + pad_x)
        y1 = min(image_height, y + height + pad_y)
        if x1 <= x0 or y1 <= y0:
            return image
        return image[y0:y1, x0:x1]

    @staticmethod
    def _center_crop(image: np.ndarray, fraction: float = 0.50) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        crop_width = max(1, int(image_width * fraction))
        crop_height = max(1, int(image_height * fraction))
        x0 = max(0, (image_width - crop_width) // 2)
        y0 = max(0, (image_height - crop_height) // 2)
        return image[y0 : y0 + crop_height, x0 : x0 + crop_width]

    def _variance_to_score(self, variance: float) -> float:
        """Map Laplacian variance to a stable 0..1 sharpness score."""

        if self.thresholds.sharpness_variance <= 0:
            return 1.0
        ratio = variance / self.thresholds.sharpness_variance
        return clamp(ratio / (ratio + 1.0))
