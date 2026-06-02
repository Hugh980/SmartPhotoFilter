from __future__ import annotations

import numpy as np

from src.detector.aesthetic import AestheticScorer


def test_aesthetic_scores_are_bounded():
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    image[:, :120, 0] = 220
    image[:, 120:, 1] = 180

    result = AestheticScorer(use_model=False).score_array(image)

    assert 0 <= result.nima_score <= 10
    assert 0 <= result.composition_score <= 10
    assert 0 <= result.color_harmony_score <= 10
    assert 0 <= result.portrait_score <= 10
    assert 0 <= result.final_score <= 10
