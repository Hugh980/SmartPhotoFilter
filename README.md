# SmartPhotoFilter

**English**: SmartPhotoFilter is a native macOS AI photo curation app for fast
batch selection. It scans photo folders, scores each image with computer-vision
signals, separates `KEEP` / `REVIEW` / `DISCARD` results, removes highly similar
burst shots, and exports organized folders plus JSON, CSV, and Markdown reports.

**中文**：SmartPhotoFilter 是一款原生 macOS AI 照片筛选工具，面向批量挑图和废片剔除场景。它会扫描照片文件夹，结合计算机视觉信号为每张图片评分，将结果划分为 `保留` / `复核` / `丢弃`，自动剔除高度相似的连拍照片，并导出分类文件夹以及 JSON、CSV、Markdown 报告。

The app is optimized for Apple Silicon Macs and uses Apple Vision where possible,
including face landmarks and image FeaturePrint similarity checks.

本项目针对 Apple Silicon（M 系列芯片）Mac 优化，并优先使用 Apple Vision 能力，包括人脸关键点检测和 FeaturePrint 相似度比对。

## Highlights / 核心亮点

- **Native desktop workflow / 原生桌面工作流**: PyQt6 desktop UI with drag-and-drop import, responsive photo grid, result table, progress feedback, and dark styling.
- **Apple Vision portrait screening / Apple Vision 人像筛选**: Screens face quality, landmarks, expression state, closed eyes, obstruction, and poor face angles.
- **Sharpness detection / 清晰度检测**: Uses Laplacian variance with optional model-backed refinement.
- **Aesthetic scoring / 美学评分**: Supports NIMA-compatible model hooks plus deterministic computer-vision fallback, so the app can still run without bundled models.
- **FeaturePrint deduplication / FeaturePrint 相似照片剔除**: Uses Apple Vision FeaturePrint distance to discard lower-scoring near-duplicates from burst sequences.
- **Visual discard tags / 直观废片标签**: Shows overlay tags in the photo grid for core discard reasons such as blur, closed eyes, obstruction, and similarity.
- **Flexible export / 灵活导出**: Supports copy, move, and report-only export modes.
- **Standalone macOS packaging / 独立 macOS 打包**: Includes `py2app` packaging, custom app icon, and ARM64 builds when using ARM64 Python.

## Workflow / 使用流程

1. **Import / 导入**: Add images or folders in the desktop app.
2. **Analyze / 分析**: Run AI-assisted scoring on every supported photo.
3. **Review / 复核**: Browse the photo grid and result table.
   - `KEEP` / `保留`: strong candidates.
   - `REVIEW` / `复核`: usable photos that need human attention.
   - `DISCARD` / `丢弃`: low-quality, expression-failed, obstructed, blurry, or redundant frames.
4. **Export / 导出**: Export organized folders and reports.

If no export path is selected in the GUI, SmartPhotoFilter defaults to:

如果用户没有手动选择导出路径，SmartPhotoFilter 会默认导出到：

```text
~/Desktop/SmartPhotoFilter-output
```

This avoids macOS packaged-app relative paths resolving into hard-to-find
temporary directories.

这样可以避免 macOS 独立应用中的相对路径落到难以查找的系统临时目录。

## Project Layout / 项目结构

```text
SmartPhotoFilter/
├── assets/                 # App icon source and generated .icns / 应用图标资源
├── docs/                   # Architecture, algorithm, and design notes / 文档
├── models/                 # Optional Core ML model location / 可选模型目录
├── scripts/                # Model/export/benchmark helpers / 辅助脚本
├── src/
│   ├── detector/           # Sharpness, face, expression, semantic, similarity, aesthetic
│   ├── gui/                # PyQt6 desktop app and widgets / 桌面界面
│   ├── models/             # Optional Core ML runtime helpers / Core ML 辅助模块
│   ├── pipeline/           # Scanning, scoring, export / 扫描、评分、导出流程
│   ├── utils/              # Image, EXIF, and platform helpers / 工具函数
│   └── main.py             # CLI and GUI entry point / 入口文件
├── tests/                  # Unit tests / 单元测试
├── pyproject.toml
└── setup.py                # py2app packaging script / macOS 打包脚本
```

## Requirements / 环境要求

- macOS 11 or newer / macOS 11 或更高版本。
- Python 3.11+.
- Apple Silicon is recommended for the packaged app / 推荐在 Apple Silicon Mac 上打包和运行。
- Optional Apple Vision dependencies are available through PyObjC packages on macOS / Apple Vision 相关能力通过 macOS 上的 PyObjC 包提供。

Core dependencies are defined in `pyproject.toml`.

核心依赖定义在 `pyproject.toml` 中。

## Development Setup / 开发环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gui,ml]"
```

Run tests / 运行测试：

```bash
pytest
```

Run the desktop app / 启动桌面应用：

```bash
python -m src.main --gui
```

Run the CLI / 使用命令行：

```bash
python -m src.main --input /path/to/photos --output /path/to/output
```

## Packaging a macOS App / 打包 macOS 应用

Install the packaging dependency / 安装打包依赖：

```bash
pip install py2app
```

Build the standalone app / 构建独立 `.app`：

```bash
rm -rf build dist
python setup.py py2app
```

The generated app will be available at / 生成的应用位于：

```text
dist/SmartPhotoFilter.app
```

Run it from Terminal / 从终端运行：

```bash
open dist/SmartPhotoFilter.app
```

Check the executable architecture / 检查可执行文件架构：

```bash
file dist/SmartPhotoFilter.app/Contents/MacOS/SmartPhotoFilter
lipo -archs dist/SmartPhotoFilter.app/Contents/MacOS/SmartPhotoFilter
```

When built with an ARM64 Python on Apple Silicon, the app executable is `arm64`.

如果使用 Apple Silicon 上的 ARM64 Python 打包，应用可执行文件会是 `arm64` 架构。

## Optional Model Artifacts / 可选模型文件

SmartPhotoFilter can run without model files by using deterministic CV fallback
scoring. Optional model artifacts can be placed in `models/`:

SmartPhotoFilter 即使没有模型文件也可以通过确定性的 CV 回退逻辑运行。可选模型文件可以放在 `models/`：

- `blur_detector.mlpackage`
- `nima_aesthetic.mlpackage`

Large model files are intentionally ignored by Git. Keep them outside the
repository or publish them as separate release assets.

大型模型文件默认不会提交到 Git。建议将它们保存在仓库外，或作为 GitHub Release 附件单独发布。

## Export Output / 导出结果

Depending on the selected export mode, SmartPhotoFilter can create:

根据导出模式，SmartPhotoFilter 可以生成：

- `keep/`
- `review/`
- `discard/`
- `report.json`
- `report.csv`
- `summary.md`

## Repository Description / 仓库简介

**English**: Native macOS AI photo curation app using Apple Vision, FeaturePrint
similarity filtering, PyQt6 grid review, and py2app packaging for Apple Silicon.

**中文**：一款原生 macOS AI 照片筛选应用，基于 Apple Vision、FeaturePrint 相似度剔除、PyQt6 网格复核界面，并支持面向 Apple Silicon 的 py2app 独立应用打包。

## License / 许可证

MIT
