"""Settings panel for thresholds and export configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.config import RuntimeConfig, Thresholds
from src.gui.styles import SPACING
from src.pipeline.exporter import ExportMode


@dataclass(frozen=True)
class GuiSettings:
    runtime: RuntimeConfig
    output_dir: str | None
    export_mode: ExportMode


class ThresholdSlider(QWidget):
    """Labeled horizontal slider with live numeric value."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        default: int,
        scale: float = 1.0,
        fmt: str = "{:.0f}",
    ) -> None:
        super().__init__()
        self._scale = scale
        self._fmt = fmt

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(SPACING.xs)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(label)
        self._label.setProperty("role", "muted")
        self._value = QLabel(fmt.format(default * scale))
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._label)
        row.addStretch()
        row.addWidget(self._value)
        root.addLayout(row)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(minimum, maximum)
        self._slider.setValue(default)
        self._slider.setMinimumHeight(44)
        self._slider.valueChanged.connect(self._on_value_changed)
        root.addWidget(self._slider)

    @property
    def value(self) -> float:
        return self._slider.value() * self._scale

    def _on_value_changed(self, value: int) -> None:
        scaled = value * self._scale
        self._value.setText(self._fmt.format(scaled))
        self.valueChanged.emit(scaled)


class SettingsPanel(QFrame):
    """Left-side app controls."""

    runRequested = pyqtSignal()
    exportRequested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("role", "panel")
        self.setMinimumWidth(300)
        self.setMaximumWidth(340)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.md, SPACING.md, SPACING.md, SPACING.md)
        root.setSpacing(SPACING.md)

        title = QLabel("SmartPhotoFilter")
        title.setProperty("role", "title")
        subtitle = QLabel("批量照片质量筛选工作台")
        subtitle.setProperty("role", "muted")
        root.addWidget(title)
        root.addWidget(subtitle)

        root.addWidget(self._section_label("检测阈值"))
        self._sharpness = ThresholdSlider("清晰度方差", 20, 300, 100)
        self._ear = ThresholdSlider("闭眼阈值 EAR", 10, 40, 20, scale=0.01, fmt="{:.2f}")
        self._aesthetic = ThresholdSlider("美感保留分", 30, 90, 60, scale=0.1, fmt="{:.1f}")
        root.addWidget(self._sharpness)
        root.addWidget(self._ear)
        root.addWidget(self._aesthetic)

        root.addWidget(self._section_label("导出设置"))
        output_row = QHBoxLayout()
        output_row.setSpacing(SPACING.sm)
        self._output = QLineEdit()
        self._output.setPlaceholderText("导出文件夹")
        browse = QPushButton("浏览")
        browse.setProperty("variant", "secondary")
        browse.clicked.connect(self._browse_output)
        output_row.addWidget(self._output)
        output_row.addWidget(browse)
        root.addLayout(output_row)

        self._mode = QComboBox()
        self._mode.addItem("复制并分类照片", ExportMode.COPY.value)
        self._mode.addItem("移动并分类照片", ExportMode.MOVE.value)
        self._mode.addItem("仅生成报告", ExportMode.REPORT_ONLY.value)
        self._mode.setCurrentIndex(self._mode.findData(ExportMode.COPY.value))
        root.addWidget(self._mode)

        root.addStretch()
        self._run = QPushButton("开始分析")
        self._run.setProperty("variant", "accent")
        self._run.clicked.connect(self.runRequested.emit)
        self._export = QPushButton("导出当前结果")
        self._export.setProperty("variant", "secondary")
        self._export.clicked.connect(self.exportRequested.emit)
        root.addWidget(self._run)
        root.addWidget(self._export)

    @property
    def settings(self) -> GuiSettings:
        thresholds = Thresholds(
            sharpness_variance=self._sharpness.value,
            eye_aspect_ratio=self._ear.value,
            aesthetic_keep_score=self._aesthetic.value,
        )
        output_text = self._output.text().strip()
        if not output_text:
            output_dir = str(Path.home() / "Desktop" / "SmartPhotoFilter-output")
        else:
            output_dir = output_text
        return GuiSettings(
            runtime=RuntimeConfig(thresholds=thresholds, use_optional_ml=True),
            output_dir=output_dir,
            export_mode=ExportMode(self._mode.currentData() or ExportMode.COPY.value),
        )

    def set_running(self, running: bool) -> None:
        self._run.setEnabled(not running)
        self._run.setText("正在分析..." if running else "开始分析")
        self._export.setEnabled(not running)

    def set_output_dir(self, path: str | Path) -> None:
        self._output.setText(str(path))

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("role", "section")
        return label

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if folder:
            self._output.setText(folder)
