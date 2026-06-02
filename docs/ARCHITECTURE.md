# Architecture

SmartPhotoFilter is split into five layers:

1. `detector`: pure scoring components for sharpness, face landmarks, expression,
   and aesthetics.
2. `pipeline`: filesystem scanning, composite decision logic, batch execution, and
   export reports.
3. `models`: optional Core ML runtime wrapper and model registry.
4. `gui`: PyQt6 desktop workspace.
5. `utils`: image loading, EXIF, and platform capability helpers.

The core pipeline is GUI-independent. The CLI and desktop app both call
`scan_inputs`, `process_batch`, and `ResultExporter`.

Optional deep model dependencies are isolated behind runtime checks. This keeps tests
and fallback development deterministic while preserving the intended Core ML and
MediaPipe integration points.
