# SmartPhotoFilter

SmartPhotoFilter is a native macOS photo curation app for fast AI-assisted batch
selection. It scans a folder of photos, scores each image with computer-vision
signals, separates keep/review/discard results, and exports organized folders plus
machine-readable reports.

The project is built for Apple Silicon Macs and uses Apple Vision where possible,
including face landmarks and image FeaturePrint similarity checks for burst-photo
deduplication.

## Highlights

- Native PyQt6 desktop workspace with drag-and-drop import, responsive photo grid,
  result table, progress feedback, and dark visual styling.
- Apple Vision-based portrait screening for face quality, landmarks, expression
  state, closed eyes, obstruction, and poor face angles.
- Sharpness detection with Laplacian variance and optional model-backed refinement.
- Aesthetic scoring with NIMA-compatible hooks plus deterministic computer-vision
  fallback, so the app can still run without bundled model files.
- Apple Vision FeaturePrint similarity filtering to remove lower-scoring near
  duplicates from burst sequences.
- Overlay discard tags in the photo grid for quick review of core failure reasons
  such as blur, closed eyes, obstruction, and similarity.
- Export modes for copying, moving, or report-only output.
- JSON, CSV, and Markdown reports for downstream review or automation.
- macOS `.app` packaging through `py2app`, including a custom app icon and
  Apple Silicon `arm64` builds when using an ARM64 Python environment.

## Screens and Workflow

1. Add images or folders in the desktop app.
2. Run analysis to score every supported photo.
3. Review the grid and result table:
   - `KEEP`: strong candidates.
   - `REVIEW`: usable but needs human attention.
   - `DISCARD`: low-quality or redundant frames.
4. Export results into organized folders and reports.

If no export path is selected in the GUI, SmartPhotoFilter defaults to:

```text
~/Desktop/SmartPhotoFilter-output
```

This avoids macOS packaged-app relative paths resolving into hard-to-find
temporary directories.

## Project Layout

```text
SmartPhotoFilter/
├── assets/                 # App icon source and generated .icns
├── docs/                   # Architecture, algorithm, and design notes
├── models/                 # Optional Core ML model location
├── scripts/                # Model/export/benchmark helper scripts
├── src/
│   ├── detector/           # Sharpness, face, expression, semantic, similarity, aesthetic
│   ├── gui/                # PyQt6 desktop app and widgets
│   ├── models/             # Optional Core ML runtime helpers
│   ├── pipeline/           # Scanning, scoring, export
│   ├── utils/              # Image, EXIF, and platform helpers
│   └── main.py             # CLI and GUI entry point
├── tests/                  # Unit tests
├── pyproject.toml
└── setup.py                # py2app packaging script
```

## Requirements

- macOS 11 or newer.
- Python 3.11+.
- Apple Silicon is recommended for the packaged app.
- Optional Apple Vision dependencies are available through PyObjC packages on macOS.

Core dependencies are defined in `pyproject.toml`.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gui,ml]"
```

Run the test suite:

```bash
pytest
```

Run the desktop app:

```bash
python -m src.main --gui
```

Run the CLI:

```bash
python -m src.main --input /path/to/photos --output /path/to/output
```

## Packaging a macOS App

Install the packaging dependency:

```bash
pip install py2app
```

Build the standalone app:

```bash
rm -rf build dist
python setup.py py2app
```

The generated app will be available at:

```text
dist/SmartPhotoFilter.app
```

Run it from Terminal:

```bash
open dist/SmartPhotoFilter.app
```

Check the executable architecture:

```bash
file dist/SmartPhotoFilter.app/Contents/MacOS/SmartPhotoFilter
lipo -archs dist/SmartPhotoFilter.app/Contents/MacOS/SmartPhotoFilter
```

When built with an ARM64 Python on Apple Silicon, the app executable is `arm64`.

## Optional Model Artifacts

SmartPhotoFilter can run without model files by using deterministic CV fallback
scoring. Optional model artifacts can be placed in `models/`:

- `blur_detector.mlpackage`
- `nima_aesthetic.mlpackage`

Large model files are intentionally ignored by Git. Keep them outside the
repository or publish them as separate release assets.

## Export Output

Depending on the selected export mode, SmartPhotoFilter can create:

- `keep/`
- `review/`
- `discard/`
- `report.json`
- `report.csv`
- `summary.md`

## Repository Description

Native macOS AI photo curation app using Apple Vision, FeaturePrint similarity
filtering, PyQt6 grid review, and py2app packaging for Apple Silicon.

## License

MIT
