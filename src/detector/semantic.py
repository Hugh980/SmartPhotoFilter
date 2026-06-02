"""Semantic pre-filter for non-photographic images.

The filter runs from cheapest to most semantic:

1. Solid/white edge background detection with local pixel statistics.
2. Apple Vision OCR text-density detection.
3. Apple Vision image classification as a final semantic fallback.

All Vision paths are intentionally fail-open. Unsupported macOS versions, missing
PyObjC bridges, unreadable images, or Vision request failures return PASS so the
regular photo-quality pipeline can continue.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from src.utils.image_utils import load_rgb_image


class SemanticDecision(str, Enum):
    """Semantic pre-filter decision."""

    PASS = "pass"
    REVIEW = "review"
    DROP = "drop"


@dataclass(frozen=True)
class SemanticClassification:
    """One Vision classification observation."""

    identifier: str
    confidence: float


@dataclass(frozen=True)
class SemanticResult:
    """Result returned by SemanticPreFilter."""

    decision: SemanticDecision
    reason: str | None = None
    classifications: tuple[SemanticClassification, ...] = ()
    used_vision: bool = False
    error: str | None = None
    text_density: float | None = None
    white_edge_ratio: float | None = None
    edge_gray_variance: float | None = None

    @property
    def should_short_circuit(self) -> bool:
        return self.decision in {SemanticDecision.DROP, SemanticDecision.REVIEW}


@dataclass(frozen=True)
class SemanticRule:
    keyword: str
    decision: SemanticDecision
    reason: str


@dataclass(frozen=True)
class EdgeBackgroundStats:
    white_ratio: float
    gray_variance: float


DEFAULT_BLOCK_RULES: tuple[SemanticRule, ...] = (
    SemanticRule("screenshot", SemanticDecision.DROP, "识别为软件截图"),
    SemanticRule("screen shot", SemanticDecision.DROP, "识别为软件截图"),
    SemanticRule("software", SemanticDecision.REVIEW, "疑似软件界面"),
    SemanticRule("document", SemanticDecision.DROP, "识别为文档"),
    SemanticRule("receipt", SemanticDecision.DROP, "识别为收据"),
    SemanticRule("barcode", SemanticDecision.DROP, "识别为条形码"),
    SemanticRule("qr code", SemanticDecision.DROP, "识别为二维码"),
    SemanticRule("text", SemanticDecision.REVIEW, "识别为纯文本内容"),
    SemanticRule("menu", SemanticDecision.REVIEW, "疑似菜单或文字页面"),
    SemanticRule("web site", SemanticDecision.REVIEW, "疑似网页截图"),
    SemanticRule("website", SemanticDecision.REVIEW, "疑似网页截图"),
)


class SemanticPreFilter:
    """Lightweight pre-filter before expensive model inference."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        text_density_threshold: float = 0.03,
        white_edge_ratio_threshold: float = 0.80,
        edge_gray_variance_threshold: float = 50.0,
        edge_width: int = 10,
        rules: tuple[SemanticRule, ...] = DEFAULT_BLOCK_RULES,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.text_density_threshold = text_density_threshold
        self.white_edge_ratio_threshold = white_edge_ratio_threshold
        self.edge_gray_variance_threshold = edge_gray_variance_threshold
        self.edge_width = edge_width
        self.rules = rules

    def analyze(self, image_path: str | Path) -> SemanticResult:
        """Run solid-background, OCR density, and Vision classification filters."""

        path = Path(image_path)

        edge_result = self._analyze_solid_background(path)
        if edge_result is not None:
            return edge_result

        text_result = self._analyze_text_density(path)
        if text_result.should_short_circuit:
            return text_result

        return self._analyze_classification(path, prior_error=text_result.error)

    def _analyze_solid_background(self, image_path: Path) -> SemanticResult | None:
        """Detect white or solid-color borders typical of product material images."""

        try:
            image = load_rgb_image(image_path, max_side=1200)
            stats = self._edge_background_stats(image, edge_width=self.edge_width)
        except Exception:
            return None

        if stats.white_ratio > self.white_edge_ratio_threshold:
            return SemanticResult(
                decision=SemanticDecision.DROP,
                reason="检测到纯色/白底(疑似商品素材)",
                white_edge_ratio=stats.white_ratio,
                edge_gray_variance=stats.gray_variance,
            )

        if stats.gray_variance < self.edge_gray_variance_threshold:
            return SemanticResult(
                decision=SemanticDecision.DROP,
                reason="检测到纯色/白底(疑似商品素材)",
                white_edge_ratio=stats.white_ratio,
                edge_gray_variance=stats.gray_variance,
            )

        return None

    def _analyze_text_density(self, image_path: Path) -> SemanticResult:
        """Detect dense text layouts using VNRecognizeTextRequest."""

        if platform.system() != "Darwin":
            return SemanticResult(
                decision=SemanticDecision.PASS,
                error="Apple Vision OCR is only available on macOS.",
            )

        try:
            density = self._text_density_with_vision(image_path)
        except Exception as exc:
            return SemanticResult(
                decision=SemanticDecision.PASS,
                error=f"Apple Vision OCR text-density detection failed: {exc}",
            )

        if density > self.text_density_threshold:
            return SemanticResult(
                decision=SemanticDecision.DROP,
                reason="包含过多文本(截图或排版图)",
                used_vision=True,
                text_density=density,
            )

        return SemanticResult(
            decision=SemanticDecision.PASS,
            used_vision=True,
            text_density=density,
        )

    def _analyze_classification(self, image_path: Path, prior_error: str | None = None) -> SemanticResult:
        """Run VNClassifyImageRequest as a semantic fallback."""

        if platform.system() != "Darwin":
            return SemanticResult(
                decision=SemanticDecision.PASS,
                error=prior_error or "Apple Vision is only available on macOS.",
            )

        try:
            classifications = self._classify_with_vision(image_path)
        except Exception as exc:
            error = f"Apple Vision semantic classification failed: {exc}"
            if prior_error:
                error = f"{prior_error}; {error}"
            return SemanticResult(decision=SemanticDecision.PASS, error=error)

        for item in classifications:
            if item.confidence < self.confidence_threshold:
                continue
            identifier = item.identifier.casefold()
            for rule in self.rules:
                if rule.keyword.casefold() in identifier:
                    return SemanticResult(
                        decision=rule.decision,
                        reason=rule.reason,
                        classifications=classifications,
                        used_vision=True,
                    )

        return SemanticResult(
            decision=SemanticDecision.PASS,
            classifications=classifications,
            used_vision=True,
        )

    @staticmethod
    def _edge_background_stats(image: np.ndarray, edge_width: int = 10) -> EdgeBackgroundStats:
        """Return white-pixel ratio and grayscale variance for image border pixels."""

        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Expected an RGB image array.")

        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Image has zero area.")

        edge = max(1, min(edge_width, height, width))
        top = image[:edge, :, :3]
        bottom = image[height - edge :, :, :3]
        left = image[:, :edge, :3]
        right = image[:, width - edge :, :3]
        border_pixels = np.concatenate(
            [
                top.reshape(-1, 3),
                bottom.reshape(-1, 3),
                left.reshape(-1, 3),
                right.reshape(-1, 3),
            ],
            axis=0,
        ).astype(np.float32)

        white_mask = np.all(border_pixels > 240.0, axis=1)
        white_ratio = float(np.mean(white_mask)) if border_pixels.size else 0.0
        gray = (
            0.2126 * border_pixels[:, 0]
            + 0.7152 * border_pixels[:, 1]
            + 0.0722 * border_pixels[:, 2]
        )
        gray_variance = float(np.var(gray)) if gray.size else 0.0
        return EdgeBackgroundStats(white_ratio=white_ratio, gray_variance=gray_variance)

    @staticmethod
    def _image_request_handler(image_path: Path):
        """Create a VNImageRequestHandler and keep its CGImage alive for the request."""

        import Quartz
        import Vision
        from CoreFoundation import CFURLCreateFromFileSystemRepresentation, kCFAllocatorDefault

        path_bytes = str(image_path).encode("utf-8")
        image_url = CFURLCreateFromFileSystemRepresentation(
            kCFAllocatorDefault,
            path_bytes,
            len(path_bytes),
            False,
        )
        if image_url is None:
            raise ValueError(f"Cannot create CFURL for image: {image_path}")

        image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
        if image_source is None:
            raise ValueError(f"Cannot create CGImageSource for image: {image_path}")

        cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if cg_image is None:
            raise ValueError(f"Cannot decode CGImage for image: {image_path}")

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
        return handler

    @classmethod
    def _text_density_with_vision(cls, image_path: Path) -> float:
        """Return total OCR bounding-box area ratio from VNRecognizeTextRequest."""

        import Vision

        request = Vision.VNRecognizeTextRequest.alloc().init()
        if hasattr(request, "setRecognitionLevel_"):
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        if hasattr(request, "setUsesLanguageCorrection_"):
            request.setUsesLanguageCorrection_(False)

        handler = cls._image_request_handler(image_path)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(error or "VNRecognizeTextRequest failed")

        total_area = 0.0
        for observation in request.results() or []:
            confidence = float(observation.confidence()) if hasattr(observation, "confidence") else 1.0
            if confidence <= 0.5:
                continue
            bbox = observation.boundingBox()
            width = max(0.0, float(bbox.size.width))
            height = max(0.0, float(bbox.size.height))
            total_area += width * height
        return min(total_area, 1.0)

    @classmethod
    def _classify_with_vision(cls, image_path: Path) -> tuple[SemanticClassification, ...]:
        """Call VNClassifyImageRequest through PyObjC and return sorted observations."""

        import Vision

        request = Vision.VNClassifyImageRequest.alloc().init()
        handler = cls._image_request_handler(image_path)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(error or "VNClassifyImageRequest failed")

        classifications: list[SemanticClassification] = []
        for observation in request.results() or []:
            identifier = str(observation.identifier())
            confidence = float(observation.confidence())
            classifications.append(
                SemanticClassification(identifier=identifier, confidence=confidence)
            )

        return tuple(sorted(classifications, key=lambda item: item.confidence, reverse=True))
