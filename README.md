# SmartPhotoFilter

English | [简体中文](#简体中文)

## English

SmartPhotoFilter is a native macOS AI photo curation app for fast batch
selection. It scans photo folders, scores each image with computer-vision
signals, separates `KEEP` / `REVIEW` / `DISCARD` results, removes highly similar
burst shots, and exports organized folders plus JSON, CSV, and Markdown reports.

The app is optimized for Apple Silicon Macs and uses Apple Vision where possible,
including face landmarks and image FeaturePrint similarity checks.

### Highlights

- Native PyQt6 desktop workflow with drag-and-drop import, responsive photo grid,
  result table, progress feedback, and dark styling.
- Apple Vision portrait screening for face quality, landmarks, expression state,
  closed eyes, obstruction, and poor face angles.
- Sharpness detection with Laplacian variance and optional model-backed refinement.
- Aesthetic scoring with NIMA-compatible hooks plus deterministic computer-vision
  fallback, so the app can still run without bundled models.
- Apple Vision FeaturePrint deduplication to discard lower-scoring near-duplicates
  from burst sequences.
- Visual discard tags in the photo grid for blur, closed eyes, obstruction, and
  similarity reasons.
- Flexible export modes for copy, move, and report-only output.
- Standalone macOS packaging through `py2app`, including a custom app icon and
  ARM64 builds when using ARM64 Python.

### Workflow

1. Add images or folders in the desktop app.
2. Run AI-assisted scoring on every supported photo.
3. Review the photo grid and result table.
4. Export organized folders and reports.

If no export path is selected in the GUI, SmartPhotoFilter defaults to:

```text
~/Desktop/SmartPhotoFilter-output
```

This avoids macOS packaged-app relative paths resolving into hard-to-find
temporary directories.

### Project Layout

```text
SmartPhotoFilter/
├── assets/
├── docs/
├── models/
├── scripts/
├── src/
├── tests/
├── pyproject.toml
└── setup.py
```

### Requirements

- macOS 11 or newer.
- Python 3.11+.
- Apple Silicon is recommended for the packaged app.
- Optional Apple Vision dependencies are available through PyObjC packages on
  macOS.

### Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gui,ml]"
```

Run tests:

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

### Packaging a macOS App

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

### Optional Model Artifacts

SmartPhotoFilter can run without model files by using deterministic CV fallback
scoring. Optional model artifacts can be placed in `models/`:

- `blur_detector.mlpackage`
- `nima_aesthetic.mlpackage`

Large model files are intentionally ignored by Git. Keep them outside the
repository or publish them as separate release assets.

### Export Output

Depending on the selected export mode, SmartPhotoFilter can create:

- `keep/`
- `review/`
- `discard/`
- `report.json`
- `report.csv`
- `summary.md`

### Repository Description

Native macOS AI photo curation app using Apple Vision, FeaturePrint similarity
filtering, PyQt6 grid review, and py2app packaging for Apple Silicon.

## 中文

[English](#english) | 简体中文

SmartPhotoFilter 是一款原生 macOS AI 照片筛选工具，面向批量挑图和废片剔除场景。它会扫描照片文件夹，结合计算机视觉信号为每张图片评分，将结果划分为 `保留` / `复核` / `丢弃`，自动剔除高度相似的连拍照片，并导出分类文件夹以及 JSON、CSV、Markdown 报告。

本项目针对 Apple Silicon（M 系列芯片）Mac 优化，并优先使用 Apple Vision 能力，包括人脸关键点检测和 FeaturePrint 相似度比对。

### 核心亮点

- 原生 PyQt6 桌面工作流，支持拖拽导入、响应式照片网格、结果表格、进度反馈和深色风格。
- 基于 Apple Vision 的人像筛选，可检测人脸质量、关键点、表情状态、闭眼、遮挡和不佳角度。
- 采用拉普拉斯方差进行清晰度检测，并支持可选模型增强。
- 美学评分支持 NIMA 兼容接口和确定性的计算机视觉回退逻辑，因此没有模型文件也能运行。
- 使用 Apple Vision FeaturePrint 相似度去除连拍中的低分近似重复照片。
- 照片网格中带有直观的丢弃标签，快速展示模糊、闭眼、遮挡和相似等原因。
- 提供复制、移动和仅报告三种导出模式。
- 通过 `py2app` 打包为独立 macOS 应用，包含自定义图标，并且在 ARM64 Python 环境下生成 ARM64 架构的应用。

### 使用流程

1. 在桌面应用中导入图片或文件夹。
2. 对所有支持的照片运行 AI 评分。
3. 在照片网格和结果表中复核。
4. 导出整理好的文件夹和报告。

如果用户没有手动选择导出路径，SmartPhotoFilter 会默认导出到：

```text
~/Desktop/SmartPhotoFilter-output
```

这样可以避免 macOS 独立应用中的相对路径落到难以查找的系统临时目录。

### 项目结构

```text
SmartPhotoFilter/
├── assets/
├── docs/
├── models/
├── scripts/
├── src/
├── tests/
├── pyproject.toml
└── setup.py
```

### 环境要求

- macOS 11 或更高版本。
- Python 3.11+。
- 推荐在 Apple Silicon Mac 上打包和运行。
- Apple Vision 相关能力通过 macOS 上的 PyObjC 包提供。

### 开发环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gui,ml]"
```

运行测试：

```bash
pytest
```

启动桌面应用：

```bash
python -m src.main --gui
```

使用命令行：

```bash
python -m src.main --input /path/to/photos --output /path/to/output
```

### 打包 macOS 应用

安装打包依赖：

```bash
pip install py2app
```

构建独立 `.app`：

```bash
rm -rf build dist
python setup.py py2app
```

生成的应用位于：

```text
dist/SmartPhotoFilter.app
```

从终端运行：

```bash
open dist/SmartPhotoFilter.app
```

检查可执行文件架构：

```bash
file dist/SmartPhotoFilter.app/Contents/MacOS/SmartPhotoFilter
lipo -archs dist/SmartPhotoFilter.app/Contents/MacOS/SmartPhotoFilter
```

如果使用 Apple Silicon 上的 ARM64 Python 打包，应用可执行文件会是 `arm64` 架构。

### 可选模型文件

SmartPhotoFilter 即使没有模型文件也可以通过确定性的 CV 回退逻辑运行。可选模型文件可以放在 `models/`：

- `blur_detector.mlpackage`
- `nima_aesthetic.mlpackage`

大型模型文件默认不会提交到 Git。建议将它们保存在仓库外，或作为 GitHub Release 附件单独发布。

### 导出结果

根据导出模式，SmartPhotoFilter 可以生成：

- `keep/`
- `review/`
- `discard/`
- `report.json`
- `report.csv`
- `summary.md`

### 仓库简介

一款原生 macOS AI 照片筛选应用，基于 Apple Vision、FeaturePrint 相似度剔除、PyQt6 网格复核界面，并支持面向 Apple Silicon 的 py2app 独立应用打包。

## License

MIT
