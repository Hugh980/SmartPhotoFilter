"""Masonry-style photo wall with non-cropped previews."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from src.pipeline.scorer import Decision, PhotoScore


MIN_COLUMNS = 3
MAX_COLUMNS = 4
MIN_COLUMN_WIDTH = 176
COLUMN_SPACING = 14
CARD_PADDING = 8
CARD_SPACING = 10
IMAGE_RADIUS = 10


class PhotoCard(QFrame):
    """One masonry card with a proportional image and compact result text."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self._source = QPixmap(path)
        self._result: PhotoScore | None = None
        self._image_width = 0

        self.setProperty("role", "photo-card")
        self.setToolTip(path)

        root = QVBoxLayout(self)
        root.setContentsMargins(CARD_PADDING, CARD_PADDING, CARD_PADDING, CARD_PADDING)
        root.setSpacing(CARD_SPACING)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setProperty("role", "photo-preview")
        root.addWidget(self._preview)

        self._caption = QLabel()
        self._caption.setWordWrap(True)
        self._caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._caption.setProperty("role", "photo-caption")
        root.addWidget(self._caption)

        self._refresh_caption()

    def apply_result(self, result: PhotoScore) -> None:
        self._result = result
        self.setProperty("decision", result.decision.value)
        self._refresh_caption()
        self._refresh_pixmap()
        self.style().unpolish(self)
        self.style().polish(self)

    def set_column_width(self, column_width: int) -> None:
        self.setFixedWidth(column_width)
        image_width = max(1, column_width - CARD_PADDING * 2)
        if image_width == self._image_width:
            return
        self._image_width = image_width
        self._refresh_pixmap()

    def estimated_height(self, column_width: int) -> int:
        image_width = max(1, column_width - CARD_PADDING * 2)
        return self._scaled_image_height(image_width) + 64

    def _refresh_caption(self) -> None:
        if self._result is None:
            self._caption.setText(f"{Path(self.path).name}\n等待分析")
            return
        self._caption.setText(
            f"{Path(self.path).name}\n"
            f"{_decision_label(self._result.decision)} | {self._result.final_score:.2f}"
        )

    def _refresh_pixmap(self) -> None:
        if self._image_width <= 0:
            return
        pixmap = _render_preview(self._source, self._image_width, self._result)
        self._preview.setPixmap(pixmap)
        self._preview.setFixedSize(pixmap.size())

    def _scaled_image_height(self, image_width: int) -> int:
        if self._source.isNull() or self._source.width() <= 0:
            return int(image_width * 0.68)
        return max(1, round(self._source.height() * image_width / self._source.width()))


class PhotoGrid(QScrollArea):
    """Scrollable masonry grid with 3-4 shortest-column layout."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("photoGrid")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._container.setObjectName("masonryContainer")
        self._root = QHBoxLayout(self._container)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(COLUMN_SPACING)
        self.setWidget(self._container)

        self._paths: list[str] = []
        self._cards: dict[str, PhotoCard] = {}
        self._columns: list[QWidget] = []
        self._column_layouts: list[QVBoxLayout] = []
        self._column_count = 0
        self._column_width = 0

    def set_paths(self, paths: list[str]) -> None:
        self._clear_cards()
        self._paths = paths
        self._cards = {path: PhotoCard(path) for path in paths}
        self._reflow(force=True)

    def set_results(self, results: list[PhotoScore]) -> None:
        for result in results:
            card = self._cards.get(result.path)
            if card is not None:
                card.apply_result(result)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reflow()

    def _reflow(self, *, force: bool = False) -> None:
        viewport_width = max(1, self.viewport().width())
        column_count = _column_count_for_width(viewport_width)
        column_width = _column_width_for(viewport_width, column_count)

        if (
            not force
            and column_count == self._column_count
            and abs(column_width - self._column_width) < 4
        ):
            return

        if column_count != self._column_count:
            self._build_columns(column_count)
        else:
            self._clear_column_layouts()

        self._column_count = column_count
        self._column_width = column_width
        self._container.setMinimumWidth(viewport_width)

        column_heights = [0 for _ in range(column_count)]
        for path in self._paths:
            card = self._cards[path]
            card.set_column_width(column_width)
            column_index = min(range(column_count), key=lambda index: column_heights[index])
            self._column_layouts[column_index].addWidget(card)
            column_heights[column_index] += card.estimated_height(column_width) + CARD_SPACING

    def _build_columns(self, column_count: int) -> None:
        self._clear_column_layouts()
        while self._root.count():
            item = self._root.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self._columns = []
        self._column_layouts = []
        for _ in range(column_count):
            column = QWidget()
            layout = QVBoxLayout(column)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(CARD_SPACING)
            layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            self._root.addWidget(column, stretch=1)
            self._columns.append(column)
            self._column_layouts.append(layout)

    def _clear_column_layouts(self) -> None:
        for layout in self._column_layouts:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)

    def _clear_cards(self) -> None:
        self._clear_column_layouts()
        for card in self._cards.values():
            card.deleteLater()
        self._cards = {}


def _column_count_for_width(width: int) -> int:
    return MAX_COLUMNS if width >= 920 else MIN_COLUMNS


def _column_width_for(width: int, column_count: int) -> int:
    available = width - COLUMN_SPACING * (column_count - 1)
    return max(MIN_COLUMN_WIDTH, available // column_count)


def _render_preview(source: QPixmap, width: int, result: PhotoScore | None) -> QPixmap:
    if source.isNull():
        canvas = QPixmap(QSize(width, int(width * 0.68)))
        canvas.fill(QColor("#1F2937"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor("#9CA3AF"))
        painter.drawText(canvas.rect(), Qt.AlignmentFlag.AlignCenter, "预览")
    else:
        scaled = source.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
        canvas = QPixmap(scaled.size())
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(0, 0, canvas.width(), canvas.height(), IMAGE_RADIUS, IMAGE_RADIUS)
        painter.setClipPath(clip)
        painter.drawPixmap(0, 0, scaled)
        painter.setClipping(False)

    painter.setPen(QColor(255, 255, 255, 42))
    painter.drawRoundedRect(0, 0, canvas.width() - 1, canvas.height() - 1, IMAGE_RADIUS, IMAGE_RADIUS)

    if result is not None and result.decision == Decision.DISCARD:
        _draw_discard_tag(painter, canvas.size(), result)

    painter.end()
    return canvas


def _draw_discard_tag(painter: QPainter, size: QSize, result: PhotoScore) -> None:
    tag = _discard_tag(result.reasons)
    font = QFont("Arial", 11)
    font.setBold(True)
    painter.setFont(font)
    metrics = QFontMetrics(font)
    tag = metrics.elidedText(tag, Qt.TextElideMode.ElideRight, size.width() - 28)
    tag_width = min(size.width() - 16, metrics.horizontalAdvance(tag) + 18)
    rect = QRect(size.width() - tag_width - 8, size.height() - 32, tag_width, 24)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(220, 38, 38, 224))
    painter.drawRoundedRect(rect, 8, 8)
    painter.setPen(QColor("#FFFFFF"))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, tag)


def _decision_label(decision: Decision) -> str:
    return {
        Decision.KEEP: "保留",
        Decision.REVIEW: "待审查",
        Decision.DISCARD: "丢弃",
    }.get(decision, decision.value)


def _discard_tag(reasons: tuple[str, ...]) -> str:
    text = " ".join(reasons).lower()
    if "相似" in text or "连拍" in text or "similar" in text:
        return "相似"
    if "包含过多文本" in text or "text" in text:
        return "过多文本"
    if "闭眼" in text or "眯眼" in text or "eye" in text:
        return "闭眼"
    if "blur" in text or "模糊" in text:
        return "模糊"
    if "遮挡" in text or "occlusion" in text:
        return "遮挡"
    if "multiple faces" in text or "多人" in text:
        return "多人"
    if "low aesthetic" in text or "美感" in text:
        return "低分"
    if not reasons:
        return "丢弃"
    return reasons[-1][:6]
