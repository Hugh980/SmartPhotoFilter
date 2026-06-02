from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from src.config import Thresholds
from src.detector.sharpness import SharpnessDetector


def test_sharpness_separates_checkerboard_from_blur(tmp_path):
    checker = np.indices((256, 256)).sum(axis=0) % 2
    sharp_array = np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)
    sharp_path = tmp_path / "sharp.png"
    blur_path = tmp_path / "blur.png"
    Image.fromarray(sharp_array, mode="RGB").save(sharp_path)
    Image.fromarray(sharp_array, mode="RGB").filter(ImageFilter.GaussianBlur(8)).save(blur_path)

    detector = SharpnessDetector(Thresholds(sharpness_variance=100.0), use_deep_model=False)
    sharp = detector.analyze(sharp_path)
    blur = detector.analyze(blur_path)

    assert sharp.laplacian_variance > blur.laplacian_variance
    assert not sharp.is_blurry
    assert blur.is_blurry


def test_sharpness_roi_rescues_bokeh_like_center_subject():
    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    subject = np.full((30, 30, 3), 128, dtype=np.uint8)
    subject[:, ::8] = 180
    subject[::8, :] = 80
    image[113:143, 113:143] = subject

    detector = SharpnessDetector(Thresholds(sharpness_variance=100.0), use_deep_model=False)
    global_variance = detector.laplacian_variance(image)
    result = detector.analyze_array(image)

    assert 0 < global_variance < 100.0
    assert result.laplacian_variance >= 100.0
    assert result.rescued_by_roi
    assert not result.is_blurry


def test_get_roi_center_crop_fallback_returns_middle_half(monkeypatch):
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    detector = SharpnessDetector(use_deep_model=False)

    monkeypatch.setattr("src.detector.sharpness.cv2", None)
    roi = detector._get_roi(image)

    assert roi.shape == (50, 60, 3)
