"""Main PyQt6 application window."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.gui.styles import SPACING, build_stylesheet
from src.gui.widgets.photo_grid import PhotoGrid
from src.gui.widgets.result_panel import ResultPanel
from src.gui.widgets.settings_panel import SettingsPanel
from src.pipeline.exporter import ResultExporter
from src.pipeline.scanner import scan_inputs
from src.pipeline.scorer import PhotoScore, process_batch
from src.utils.platform_utils import detect_runtime_capabilities


class DropZone(QFrame):
    """Drag target and import entry point."""

    pathsDropped = pyqtSignal(list)
    browseRequested = pyqtSignal()
    folderBrowseRequested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("role", "dropzone")
        self.setProperty("dragging", "false")
        self.setAcceptDrops(True)
        self.setMinimumHeight(180)

        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.setSpacing(SPACING.sm)

        title = QLabel("拖入照片或文件夹")
        title.setProperty("role", "section")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("支持 JPG、PNG、HEIC、TIFF、BMP 和 WebP")
        subtitle.setProperty("role", "muted")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)

        button_row = QHBoxLayout()
        button_row.setSpacing(SPACING.sm)
        photo_button = QPushButton("选择照片")
        photo_button.setProperty("variant", "secondary")
        photo_button.clicked.connect(self.browseRequested.emit)
        folder_button = QPushButton("选择文件夹")
        folder_button.setProperty("variant", "secondary")
        folder_button.clicked.connect(self.folderBrowseRequested.emit)
        button_row.addWidget(photo_button)
        button_row.addWidget(folder_button)
        root.addLayout(button_row)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_dragging(True)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_dragging(False)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._set_dragging(False)
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        if paths:
            self.pathsDropped.emit(paths)

    def _set_dragging(self, dragging: bool) -> None:
        self.setProperty("dragging", "true" if dragging else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class ProcessingThread(QThread):
    """Runs the scoring pipeline without freezing the GUI."""

    progressChanged = pyqtSignal(int, int, object)
    completed = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, paths: list[str], settings) -> None:
        super().__init__()
        self._paths = paths
        self._settings = settings

    def run(self) -> None:
        try:
            assets = scan_inputs(self._paths)

            def progress(done: int, total: int, result: PhotoScore) -> None:
                self.progressChanged.emit(done, total, result)

            results = process_batch(assets, config=self._settings.runtime, progress=progress)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(results)


class MainWindow(QMainWindow):
    """SmartPhotoFilter desktop workspace."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SmartPhotoFilter")
        self.setMinimumSize(1180, 760)
        self.resize(1320, 820)

        self._paths: list[str] = []
        self._results: list[PhotoScore] = []
        self._worker: ProcessingThread | None = None

        self._build_menu()
        self._build_layout()
        self._build_status_bar()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")

        open_action = QAction("打开照片", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._browse)
        file_menu.addAction(open_action)

        open_folder_action = QAction("打开文件夹", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self._browse_folder)
        file_menu.addAction(open_folder_action)

        export_action = QAction("导出当前结果", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction("退出", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_layout(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(SPACING.md, SPACING.md, SPACING.md, SPACING.md)
        root.setSpacing(SPACING.md)

        self._settings = SettingsPanel()
        self._settings.runRequested.connect(self._run)
        self._settings.exportRequested.connect(self._export)
        root.addWidget(self._settings)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        work = QWidget()
        work_layout = QVBoxLayout(work)
        work_layout.setContentsMargins(0, 0, 0, 0)
        work_layout.setSpacing(SPACING.md)

        self._drop = DropZone()
        self._drop.pathsDropped.connect(self._load_paths)
        self._drop.browseRequested.connect(self._browse)
        self._drop.folderBrowseRequested.connect(self._browse_folder)
        self._grid = PhotoGrid()
        work_layout.addWidget(self._drop)
        work_layout.addWidget(self._grid, stretch=1)

        self._results_panel = ResultPanel()
        splitter.addWidget(work)
        splitter.addWidget(self._results_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

        self.setCentralWidget(central)

    def _build_status_bar(self) -> None:
        self._progress = QProgressBar()
        self._progress.setFixedWidth(260)
        self._progress.setValue(0)
        self._status = QLabel(_capability_label())
        self._status.setProperty("role", "muted")
        bar = QStatusBar()
        bar.addWidget(self._status, stretch=1)
        bar.addPermanentWidget(self._progress)
        self.setStatusBar(bar)

    def _browse(self) -> None:
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setWindowTitle("选择照片")
        dialog.setNameFilter("图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.heic *.heif *.webp)")
        if dialog.exec():
            self._load_paths(dialog.selectedFiles())

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择照片文件夹")
        if folder:
            self._load_paths([folder])

    def _load_paths(self, paths: list[str]) -> None:
        assets = scan_inputs(paths)
        self._paths = [str(asset.path) for asset in assets]
        self._results = []
        self._grid.set_paths(self._paths)
        self._results_panel.clear()
        self._progress.setValue(0)
        self._status.setText(f"已载入 {len(self._paths)} 张图片")
        if self._paths and self._settings.settings.output_dir is None:
            default_output = Path(tempfile.gettempdir()) / "SmartPhotoFilter-output"
            self._settings.set_output_dir(default_output)

    def _run(self) -> None:
        if not self._paths:
            QMessageBox.information(self, "SmartPhotoFilter", "请先选择照片或文件夹。")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._results = []
        self._settings.set_running(True)
        self._progress.setValue(0)
        self._results_panel.set_status("正在分析...")
        self._status.setText("正在分析照片")

        self._worker = ProcessingThread(self._paths, self._settings.settings)
        self._worker.progressChanged.connect(self._on_progress)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, done: int, total: int, result: PhotoScore) -> None:
        pct = int(done / total * 100) if total else 0
        self._progress.setValue(pct)
        self._status.setText(f"正在分析 {done}/{total}")
        self._grid.set_results([result])

    def _on_completed(self, results: list[PhotoScore]) -> None:
        self._results = results
        self._settings.set_running(False)
        self._progress.setValue(100 if results else 0)
        self._grid.set_results(results)
        self._results_panel.set_results(results)
        self._results_panel.set_status("分析完成")
        self._status.setText(f"完成：{len(results)} 张图片")

    def _on_failed(self, message: str) -> None:
        self._settings.set_running(False)
        self._results_panel.set_status("分析失败")
        self._status.setText("分析失败")
        QMessageBox.critical(self, "SmartPhotoFilter", message)

    def _export(self) -> None:
        if not self._results:
            QMessageBox.information(self, "SmartPhotoFilter", "请先完成分析再导出。")
            return
        settings = self._settings.settings
        output = settings.output_dir
        if output is None:
            output = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
            if not output:
                return
        summary = ResultExporter().export(self._results, output, mode=settings.export_mode)
        self._status.setText(f"已导出到 {summary.output_dir}")
        QMessageBox.information(self, "SmartPhotoFilter", f"导出完成：\n{summary.output_dir}")


def _capability_label() -> str:
    caps = detect_runtime_capabilities()
    if caps.coreml_available and caps.system == "Darwin":
        return "Core ML 已就绪"
    if caps.vision_available:
        return "Apple Vision 已就绪"
    return "CPU 兼容模式"


def create_app(argv: list[str] | None = None) -> tuple[QApplication, MainWindow]:
    app = QApplication(argv or sys.argv)
    app.setApplicationName("SmartPhotoFilter")
    app.setFont(QFont("Arial", 13))
    app.setStyleSheet(build_stylesheet())
    window = MainWindow()
    return app, window
