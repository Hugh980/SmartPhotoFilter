"""Result summary panel."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.gui.styles import SPACING
from src.pipeline.scorer import Decision, PhotoScore


class Metric(QFrame):
    """Small metric cell for result counts."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.setProperty("role", "panel")
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        root.setSpacing(SPACING.xs)
        self._value = QLabel("0")
        self._value.setProperty("role", "metric")
        text = QLabel(label)
        text.setProperty("role", "muted")
        root.addWidget(self._value)
        root.addWidget(text)

    def set_value(self, value: int) -> None:
        self._value.setText(str(value))


class ResultPanel(QFrame):
    """Right-side summary and sortable table."""

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("role", "panel")
        self.setMinimumWidth(360)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.md, SPACING.md, SPACING.md, SPACING.md)
        root.setSpacing(SPACING.md)

        title = QLabel("筛选结果")
        title.setProperty("role", "section")
        root.addWidget(title)

        metrics = QGridLayout()
        metrics.setSpacing(SPACING.sm)
        self._metrics = {
            "total": Metric("总计"),
            Decision.KEEP: Metric("保留"),
            Decision.REVIEW: Metric("待审查"),
            Decision.DISCARD: Metric("丢弃"),
        }
        metrics.addWidget(self._metrics["total"], 0, 0)
        metrics.addWidget(self._metrics[Decision.KEEP], 0, 1)
        metrics.addWidget(self._metrics[Decision.REVIEW], 1, 0)
        metrics.addWidget(self._metrics[Decision.DISCARD], 1, 1)
        root.addLayout(metrics)

        self._status = QLabel("就绪")
        self._status.setProperty("role", "muted")
        root.addWidget(self._status)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["结果", "评分", "文件", "原因"])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, stretch=1)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_results(self, results: list[PhotoScore]) -> None:
        counts = {
            decision: sum(1 for result in results if result.decision == decision)
            for decision in Decision
        }
        self._metrics["total"].set_value(len(results))
        self._metrics[Decision.KEEP].set_value(counts[Decision.KEEP])
        self._metrics[Decision.REVIEW].set_value(counts[Decision.REVIEW])
        self._metrics[Decision.DISCARD].set_value(counts[Decision.DISCARD])

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(results))
        for row, result in enumerate(results):
            values = [
                _decision_label(result.decision),
                f"{result.final_score:.2f}",
                Path(result.path).name,
                _reasons_label(result.reasons),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(result.path if column == 2 else value)
                self._table.setItem(row, column, item)
        self._table.setSortingEnabled(True)

    def clear(self) -> None:
        self.set_results([])
        self._status.setText("就绪")


def _decision_label(decision: Decision) -> str:
    return {
        Decision.KEEP: "保留",
        Decision.REVIEW: "待审查",
        Decision.DISCARD: "丢弃",
    }.get(decision, decision.value)


def _reasons_label(reasons: tuple[str, ...]) -> str:
    translated = []
    for reason in reasons:
        if reason.startswith("blur variance"):
            translated.append(reason.replace("blur variance", "模糊方差"))
        elif reason == "expression penalty":
            translated.append("表情扣分")
        elif reason == "multiple faces":
            translated.append("检测到多人脸")
        elif reason.startswith("aesthetic review band"):
            translated.append(reason.replace("aesthetic review band", "美感待审分"))
        elif reason.startswith("low aesthetic"):
            translated.append(reason.replace("low aesthetic", "美感分偏低"))
        elif reason.startswith("aesthetic"):
            translated.append(reason.replace("aesthetic", "美感分"))
        else:
            translated.append(reason)
    return "；".join(translated)
