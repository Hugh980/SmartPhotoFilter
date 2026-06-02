# Model Artifacts

This directory is reserved for optional Core ML packages:

- `blur_detector.mlpackage`: EfficientNet-B0 or similar blur detector.
- `nima_aesthetic.mlpackage`: MobileNetV3 NIMA aesthetic scorer.

The application does not require these files for local development or tests. When
they are missing, SmartPhotoFilter uses deterministic OpenCV/Pillow/Numpy fallback
scoring.
