# Algorithm Notes

## Sharpness

Stage 1 uses Laplacian variance on luminance. Images below the configured variance
threshold are treated as blurred. If `blur_detector.mlpackage` exists and Core ML can
load it, the model score refines the final blur decision.

## Expression

MediaPipe Face Mesh landmarks are used when available. Eye Aspect Ratio below the
threshold adds a strong closed-eye penalty. Mouth Aspect Ratio and gaze offset add
smaller penalties. Images without landmarks are treated as expression-neutral.

## Aesthetic Score

The intended model hook is NIMA on a MobileNetV3 backbone. Without the model, the
fallback scorer combines:

- exposure balance
- luminance contrast
- luminance entropy
- color saturation
- rule-of-thirds and balance estimate
- portrait placement and size when a face is detected

The final aesthetic score follows the project weighting:

```text
0.40 * NIMA + 0.25 * composition + 0.15 * color + 0.20 * portrait
```

## Decision Rules

- Discard: blurred image or expression score below the discard threshold.
- Review: multiple faces or aesthetic score in the review band.
- Keep: aesthetic score above the keep threshold and no hard failure.
