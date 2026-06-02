from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.detector.semantic import SemanticDecision, SemanticPreFilter
from src.pipeline.scorer import Decision, PhotoScorer


def test_semantic_prefilter_fails_open_on_unsupported_path(tmp_path):
    result = SemanticPreFilter().analyze(tmp_path / "missing.png")

    assert result.decision == SemanticDecision.PASS


def test_semantic_prefilter_drops_white_edge_product_image(tmp_path):
    path = tmp_path / "product.png"
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    image[35:85, 35:85] = [40, 80, 160]
    Image.fromarray(image, mode="RGB").save(path)

    result = SemanticPreFilter().analyze(path)

    assert result.decision == SemanticDecision.DROP
    assert result.reason == "检测到纯色/白底(疑似商品素材)"
    assert result.white_edge_ratio is not None
    assert result.white_edge_ratio > 0.80


def test_semantic_prefilter_drops_dense_text(monkeypatch, tmp_path):
    path = tmp_path / "dashboard.png"
    rng = np.random.default_rng(7)
    image = rng.integers(20, 220, size=(120, 120, 3), dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)

    monkeypatch.setattr(SemanticPreFilter, "_text_density_with_vision", lambda *_args: 0.08)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("classification should not run after dense text drop")

    monkeypatch.setattr(SemanticPreFilter, "_classify_with_vision", fail_if_called)

    result = SemanticPreFilter().analyze(path)

    assert result.decision == SemanticDecision.DROP
    assert result.reason == "包含过多文本(截图或排版图)"
    assert result.text_density == 0.08


def test_vision_text_density_uses_accurate_mode_and_filters_low_confidence(monkeypatch, tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (24, 24), color="white").save(path)

    class FakeSize:
        def __init__(self, width: float, height: float) -> None:
            self.width = width
            self.height = height

    class FakeBBox:
        def __init__(self, width: float, height: float) -> None:
            self.size = FakeSize(width, height)

    class FakeObservation:
        def __init__(self, confidence: float, width: float, height: float) -> None:
            self._confidence = confidence
            self._bbox = FakeBBox(width, height)

        def confidence(self) -> float:
            return self._confidence

        def boundingBox(self) -> FakeBBox:
            return self._bbox

    class FakeRequest:
        last = None

        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            FakeRequest.last = self
            self.recognition_level = None
            self.uses_language_correction = None
            return self

        def setRecognitionLevel_(self, value):
            self.recognition_level = value

        def setUsesLanguageCorrection_(self, value):
            self.uses_language_correction = value

        def results(self):
            return [
                FakeObservation(0.3, 0.9, 0.9),
                FakeObservation(0.5, 0.7, 0.7),
                FakeObservation(0.8, 0.2, 0.3),
            ]

    class FakeHandler:
        def performRequests_error_(self, _requests, _error):
            return True, None

    fake_vision = SimpleNamespace(
        VNRecognizeTextRequest=FakeRequest,
        VNRequestTextRecognitionLevelAccurate="accurate",
    )
    monkeypatch.setitem(sys.modules, "Vision", fake_vision)
    monkeypatch.setattr(
        SemanticPreFilter,
        "_image_request_handler",
        classmethod(lambda cls, _path: FakeHandler()),
    )

    density = SemanticPreFilter._text_density_with_vision(path)

    assert density == 0.06
    assert FakeRequest.last.recognition_level == "accurate"
    assert FakeRequest.last.uses_language_correction is False


def test_photo_scorer_short_circuits_semantic_drop(monkeypatch, tmp_path, capsys):
    from src.detector.semantic import SemanticResult

    path = tmp_path / "screenshot.png"
    Image.new("RGB", (24, 24), color="white").save(path)

    def fake_analyze(_self, _path):
        return SemanticResult(
            decision=SemanticDecision.DROP,
            reason="识别为软件截图",
            used_vision=True,
        )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("expensive detector should not be called")

    monkeypatch.setattr(SemanticPreFilter, "analyze", fake_analyze)
    scorer = PhotoScorer(use_optional_ml=False)
    monkeypatch.setattr(scorer.sharpness, "analyze_array", fail_if_called)
    monkeypatch.setattr(scorer.face, "detect_array", fail_if_called)
    monkeypatch.setattr(scorer.aesthetic, "score_array", fail_if_called)

    result = scorer.score_path(path)

    assert result.decision == Decision.DISCARD
    assert result.reasons == ("识别为软件截图",)
    assert result.semantic is not None
    assert result.semantic.decision == SemanticDecision.DROP
    debug = capsys.readouterr().out
    assert "[DEBUG] File: screenshot.png" in debug
    assert "Vision Faces: 0" in debug
    assert "MP Faces: 0" in debug
    assert "EAR: None" in debug
    assert "NIMA: 0.0" in debug
