"""Composite photo scoring and filtering decisions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

from src.config import RuntimeConfig, Thresholds
from src.detector.aesthetic import AestheticResult, AestheticScorer
from src.detector.expression import ExpressionDecision, ExpressionResult, ExpressionScorer
from src.detector.face import FaceDetectionResult, FaceDetector
from src.detector.semantic import SemanticDecision, SemanticPreFilter, SemanticResult
from src.detector.sharpness import SharpnessDetector, SharpnessResult
from src.detector.similarity import feature_print_distance, precompute_feature_prints
from src.pipeline.scanner import ImageAsset
from src.utils.image_utils import load_rgb_image
from src.utils.platform_utils import recommended_workers


SIMILARITY_DISCARD_REASON = "相似照片/连拍筛选"


class Decision(str, Enum):
    KEEP = "keep"
    REVIEW = "review"
    DISCARD = "discard"


@dataclass(frozen=True)
class PhotoScore:
    """Full result for one image."""

    path: str
    decision: Decision
    final_score: float
    reasons: tuple[str, ...]
    sharpness: SharpnessResult
    face: FaceDetectionResult
    expression: ExpressionResult
    aesthetic: AestheticResult
    semantic: SemanticResult | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload


ProgressCallback = Callable[[int, int, PhotoScore], None]


class PhotoScorer:
    """Runs all detectors and applies product decisions."""

    def __init__(self, thresholds: Thresholds = Thresholds(), use_optional_ml: bool = True) -> None:
        self.thresholds = thresholds
        self.semantic = SemanticPreFilter()
        self.sharpness = SharpnessDetector(thresholds, use_deep_model=use_optional_ml)
        self.face = FaceDetector()
        self.expression = ExpressionScorer(thresholds)
        self.aesthetic = AestheticScorer(use_model=use_optional_ml)

    def score_path(self, path: str | Path) -> PhotoScore:
        image = load_rgb_image(path, max_side=1600)
        semantic = self.semantic.analyze(path)
        if semantic.should_short_circuit:
            return self._semantic_short_circuit_score(path, semantic)

        sharpness = self.sharpness.analyze_array(image, file_path=str(path))
        face = self.face.detect_array(image, file_path=str(path))
        expression = self.expression.score(face)
        if expression.decision == ExpressionDecision.DROP:
            return self._expression_short_circuit_score(
                path=path,
                sharpness=sharpness,
                face=face,
                expression=expression,
                semantic=semantic,
            )

        aesthetic = self.aesthetic.score_array(image, face_detection=face)
        decision, reasons = self._decide(sharpness, face, expression, aesthetic)
        self._debug_print(path, face, expression, aesthetic)
        return PhotoScore(
            path=str(path),
            decision=decision,
            final_score=round(aesthetic.final_score, 3),
            reasons=tuple(reasons),
            sharpness=sharpness,
            face=face,
            expression=expression,
            aesthetic=aesthetic,
            semantic=semantic,
        )

    def close(self) -> None:
        self.face.close()

    def _decide(
        self,
        sharpness: SharpnessResult,
        face: FaceDetectionResult,
        expression: ExpressionResult,
        aesthetic: AestheticResult,
    ) -> tuple[Decision, list[str]]:
        reasons: list[str] = []

        if sharpness.is_blurry:
            reasons.append(f"blur variance {sharpness.laplacian_variance:.1f}")
            return Decision.DISCARD, reasons

        if expression.score < self.thresholds.expression_discard_score:
            reasons.append(expression.reason or "expression penalty")
            return Decision.DISCARD, reasons

        if expression.decision == ExpressionDecision.REVIEW:
            if expression.reason:
                reasons.append(expression.reason)
            return Decision.REVIEW, reasons

        if face.face_count > 1:
            reasons.append("multiple faces")
            return Decision.REVIEW, reasons

        if aesthetic.final_score >= self.thresholds.aesthetic_keep_score:
            reasons.append(f"aesthetic {aesthetic.final_score:.1f}")
            return Decision.KEEP, reasons

        if aesthetic.final_score >= self.thresholds.aesthetic_review_score:
            reasons.append(f"aesthetic review band {aesthetic.final_score:.1f}")
            return Decision.REVIEW, reasons

        reasons.append(f"low aesthetic {aesthetic.final_score:.1f}")
        return Decision.DISCARD, reasons

    @staticmethod
    def _semantic_short_circuit_score(path: str | Path, semantic: SemanticResult) -> PhotoScore:
        decision = (
            Decision.DISCARD
            if semantic.decision == SemanticDecision.DROP
            else Decision.REVIEW
        )
        reason = semantic.reason or "semantic pre-filter"
        face = FaceDetectionResult(file_path=str(path), faces=[], used_vision=False)
        expression = ExpressionResult(
            decision=ExpressionDecision.PASS,
            score=0.0,
            eyes_closed=False,
            mouth_open=False,
            gaze_off_center=False,
            eye_aspect_ratio=None,
            mouth_aspect_ratio=None,
            notes=("Semantic pre-filter short-circuited expensive detectors.",),
        )
        aesthetic = AestheticResult(
            nima_score=0.0,
            composition_score=0.0,
            color_harmony_score=0.0,
            portrait_score=0.0,
            final_score=0.0,
            used_model=False,
        )
        PhotoScorer._debug_print(path, face, expression, aesthetic)
        return PhotoScore(
            path=str(path),
            decision=decision,
            final_score=0.0,
            reasons=(reason,),
            sharpness=SharpnessResult(
                file_path=str(path),
                laplacian_variance=0.0,
                blur_score=0.0,
                is_blurry=False,
                threshold=0.0,
            ),
            face=face,
            expression=expression,
            aesthetic=aesthetic,
            semantic=semantic,
        )

    @staticmethod
    def _expression_short_circuit_score(
        path: str | Path,
        sharpness: SharpnessResult,
        face: FaceDetectionResult,
        expression: ExpressionResult,
        semantic: SemanticResult | None,
    ) -> PhotoScore:
        aesthetic = AestheticResult(
            nima_score=0.0,
            composition_score=0.0,
            color_harmony_score=0.0,
            portrait_score=0.0,
            final_score=0.0,
            used_model=False,
        )
        PhotoScorer._debug_print(path, face, expression, aesthetic)
        return PhotoScore(
            path=str(path),
            decision=Decision.DISCARD,
            final_score=0.0,
            reasons=(expression.reason or "expression drop",),
            sharpness=sharpness,
            face=face,
            expression=expression,
            aesthetic=aesthetic,
            semantic=semantic,
        )

    @staticmethod
    def _debug_print(
        path: str | Path,
        face: FaceDetectionResult,
        expression: ExpressionResult,
        aesthetic: AestheticResult,
    ) -> None:
        path = Path(path)
        print(
            f"[DEBUG] File: {path.name} | Vision Faces: {face.vision_face_count} | "
            f"MP Faces: {face.face_count} | EAR: {expression.eye_aspect_ratio} | "
            f"NIMA: {aesthetic.nima_score}"
        )


def process_batch(
    assets: Iterable[ImageAsset | str | Path],
    config: RuntimeConfig = RuntimeConfig(),
    progress: ProgressCallback | None = None,
) -> list[PhotoScore]:
    """Score images concurrently and return stable path-sorted results."""

    paths = [asset.path if isinstance(asset, ImageAsset) else Path(asset) for asset in assets]
    total = len(paths)
    if total == 0:
        return []

    workers = recommended_workers(config.max_workers)
    results: list[PhotoScore] = []

    def run_one(path: Path) -> PhotoScore:
        scorer = PhotoScorer(config.thresholds, use_optional_ml=config.use_optional_ml)
        try:
            return scorer.score_path(path)
        finally:
            scorer.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_one, path): path for path in paths}
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            result = future.result()
            results.append(result)
            if progress is not None:
                progress(completed, total, result)

    sorted_results = sorted(results, key=lambda result: result.path)
    return apply_similarity_filter(
        sorted_results,
        distance_threshold=config.thresholds.similarity_distance,
        max_workers=workers,
    )


def apply_similarity_filter(
    results: list[PhotoScore],
    *,
    distance_threshold: float,
    max_workers: int = 1,
) -> list[PhotoScore]:
    """Discard lower-scoring near-duplicates among KEEP/REVIEW results."""

    if distance_threshold <= 0:
        return results

    candidates = [result for result in results if result.decision in {Decision.KEEP, Decision.REVIEW}]
    if len(candidates) < 2:
        return results

    feature_prints = precompute_feature_prints(
        [result.path for result in candidates],
        max_workers=max_workers,
    )
    candidates = [result for result in candidates if result.path in feature_prints]
    if len(candidates) < 2:
        return results

    parent = {result.path: result.path for result in candidates}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for index, first in enumerate(candidates):
        first_print = feature_prints[first.path]
        for second in candidates[index + 1 :]:
            distance = feature_print_distance(first_print, feature_prints[second.path])
            if distance is not None and distance < distance_threshold:
                union(first.path, second.path)

    groups: dict[str, list[PhotoScore]] = {}
    for result in candidates:
        groups.setdefault(find(result.path), []).append(result)

    discard_paths: set[str] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        winner = max(
            group,
            key=lambda result: (
                result.final_score,
                result.decision == Decision.KEEP,
                result.path,
            ),
        )
        discard_paths.update(result.path for result in group if result.path != winner.path)

    if not discard_paths:
        return results

    updated: list[PhotoScore] = []
    for result in results:
        if result.path not in discard_paths:
            updated.append(result)
            continue
        reasons = result.reasons
        if SIMILARITY_DISCARD_REASON not in reasons:
            reasons = (*reasons, SIMILARITY_DISCARD_REASON)
        updated.append(replace(result, decision=Decision.DISCARD, reasons=reasons))
    return updated
