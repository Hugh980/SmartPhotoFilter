from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from src.config import RuntimeConfig, Thresholds
from src.detector.aesthetic import AestheticResult
from src.detector.expression import ExpressionDecision, ExpressionResult
from src.detector.face import FaceDetectionResult
from src.detector.sharpness import SharpnessResult
from src.pipeline.exporter import ExportMode, ResultExporter
from src.pipeline.scanner import scan_inputs
from src.pipeline import scorer as scorer_module
from src.pipeline.scorer import (
    SIMILARITY_DISCARD_REASON,
    Decision,
    PhotoScore,
    PhotoScorer,
    apply_similarity_filter,
    process_batch,
)


def test_pipeline_scans_scores_and_exports_reports(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    checker = np.indices((160, 160)).sum(axis=0) % 2
    sharp_array = np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)
    Image.fromarray(sharp_array, mode="RGB").save(input_dir / "sharp.png")
    Image.fromarray(sharp_array, mode="RGB").filter(ImageFilter.GaussianBlur(10)).save(
        input_dir / "blur.png"
    )

    assets = scan_inputs([input_dir])
    config = RuntimeConfig(
        thresholds=Thresholds(
            sharpness_variance=100.0,
            aesthetic_keep_score=0.1,
            aesthetic_review_score=0.0,
        ),
        max_workers=1,
        use_optional_ml=False,
    )
    results = process_batch(assets, config=config)

    assert len(results) == 2
    assert {result.decision for result in results} >= {Decision.KEEP, Decision.DISCARD}

    summary = ResultExporter().export(results, output_dir, ExportMode.COPY)

    assert summary.total == 2
    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.csv").exists()
    assert (output_dir / "summary.md").exists()


def test_pipeline_short_circuits_expression_drop_before_aesthetic(monkeypatch, tmp_path, capsys):
    path = tmp_path / "portrait.png"
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    image[24:40, 24:40] = 40
    Image.fromarray(image, mode="RGB").save(path)

    scorer = _closed_eye_scorer(monkeypatch, path)

    result = scorer.score_path(path)

    assert result.decision == Decision.DISCARD
    assert result.reasons == ("闭眼/眯眼 (表情管理失败)",)
    assert result.final_score == 0.0
    assert result.aesthetic.nima_score == 0.0
    debug = capsys.readouterr().out
    assert "[DEBUG] File: portrait.png" in debug
    assert "EAR: 0.18" in debug
    assert "NIMA: 0.0" in debug


def test_similarity_filter_discards_lower_scoring_near_duplicate(monkeypatch, tmp_path):
    first = str(tmp_path / "first.jpg")
    second = str(tmp_path / "second.jpg")
    third = str(tmp_path / "third.jpg")

    class FakeFeaturePrint:
        def __init__(self, path: str) -> None:
            self.path = path

    def fake_precompute(paths, max_workers):
        return {path: FakeFeaturePrint(path) for path in paths}

    distances = {
        frozenset({first, second}): 0.2,
        frozenset({first, third}): 0.9,
        frozenset({second, third}): 0.9,
    }

    def fake_distance(left, right):
        return distances[frozenset({left.path, right.path})]

    monkeypatch.setattr(scorer_module, "precompute_feature_prints", fake_precompute)
    monkeypatch.setattr(scorer_module, "feature_print_distance", fake_distance)

    results = apply_similarity_filter(
        [
            _photo_score(first, Decision.KEEP, 8.0),
            _photo_score(second, Decision.REVIEW, 6.0),
            _photo_score(third, Decision.KEEP, 7.0),
        ],
        distance_threshold=0.35,
        max_workers=2,
    )
    by_path = {result.path: result for result in results}

    assert by_path[first].decision == Decision.KEEP
    assert by_path[second].decision == Decision.DISCARD
    assert SIMILARITY_DISCARD_REASON in by_path[second].reasons
    assert by_path[third].decision == Decision.KEEP


def _photo_score(path: str, decision: Decision, final_score: float) -> PhotoScore:
    return PhotoScore(
        path=path,
        decision=decision,
        final_score=final_score,
        reasons=(),
        sharpness=SharpnessResult(
            file_path=path,
            laplacian_variance=200.0,
            blur_score=0.9,
            is_blurry=False,
            threshold=100.0,
        ),
        face=FaceDetectionResult(file_path=path, faces=[], used_vision=False),
        expression=ExpressionResult(
            decision=ExpressionDecision.PASS,
            score=0.0,
            eyes_closed=False,
            mouth_open=False,
            gaze_off_center=False,
            eye_aspect_ratio=None,
            mouth_aspect_ratio=None,
        ),
        aesthetic=AestheticResult(
            nima_score=final_score,
            composition_score=final_score,
            color_harmony_score=final_score,
            portrait_score=final_score,
            final_score=final_score,
            used_model=False,
        ),
    )


def _closed_eye_scorer(monkeypatch, path) -> PhotoScorer:
    scorer = PhotoScorer(use_optional_ml=False)
    monkeypatch.setattr(
        scorer.semantic,
        "analyze",
        lambda _path: type("PassSemantic", (), {"should_short_circuit": False})(),
    )
    monkeypatch.setattr(
        scorer.sharpness,
        "analyze_array",
        lambda *_args, **_kwargs: SharpnessResult(
            file_path=str(path),
            laplacian_variance=250.0,
            blur_score=0.9,
            is_blurry=False,
            threshold=100.0,
        ),
    )
    monkeypatch.setattr(
        scorer.face,
        "detect_array",
        lambda *_args, **_kwargs: FaceDetectionResult(file_path=str(path), faces=[], used_vision=False),
    )
    monkeypatch.setattr(
        scorer.expression,
        "score",
        lambda _face: ExpressionResult(
            decision=ExpressionDecision.DROP,
            score=-1.0,
            eyes_closed=True,
            mouth_open=False,
            gaze_off_center=False,
            eye_aspect_ratio=0.18,
            mouth_aspect_ratio=None,
            reason="闭眼/眯眼 (表情管理失败)",
        ),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("aesthetic model should not run after expression drop")

    monkeypatch.setattr(scorer.aesthetic, "score_array", fail_if_called)
    return scorer
