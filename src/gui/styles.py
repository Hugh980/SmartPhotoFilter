"""Design tokens and QSS theme for the desktop app."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Colors:
    primary: str = "#38BDF8"
    on_primary: str = "#061016"
    secondary: str = "#A78BFA"
    accent: str = "#10B981"
    background: str = "#0B0D10"
    foreground: str = "#F8FAFC"
    surface: str = "#15171B"
    muted: str = "#22262D"
    muted_foreground: str = "#9CA3AF"
    border: str = "#30343B"
    destructive: str = "#F43F5E"
    warning: str = "#F59E0B"
    ring: str = "#22D3EE"


@dataclass(frozen=True)
class Spacing:
    xs: int = 4
    sm: int = 8
    md: int = 16
    lg: int = 24
    xl: int = 32


@dataclass(frozen=True)
class Typography:
    family: str = "Arial"
    xs: int = 12
    sm: int = 13
    base: int = 15
    lg: int = 18
    xl: int = 24


COLORS = Colors()
SPACING = Spacing()
TYPOGRAPHY = Typography()


def build_stylesheet(c: Colors = COLORS, t: Typography = TYPOGRAPHY) -> str:
    """Return the QSS stylesheet following the UI UX Pro MAX design baseline."""

    return f"""
    QWidget {{
        background: {c.background};
        color: {c.foreground};
        font-family: {t.family};
        font-size: {t.base}px;
    }}

    QMainWindow, QMenuBar {{
        background: {c.background};
    }}

    QLabel, QCheckBox {{
        background: transparent;
    }}

    QMenuBar {{
        border-bottom: 1px solid {c.border};
        padding: 4px 8px;
    }}

    QMenuBar::item:selected, QMenu::item:selected {{
        background: {c.muted};
        border-radius: 6px;
    }}

    QMenu {{
        background: {c.surface};
        border: 1px solid {c.border};
        padding: 6px;
    }}

    QFrame[role="panel"], QGroupBox {{
        background: {c.surface};
        border: 1px solid {c.border};
        border-radius: 8px;
    }}

    QFrame[role="dropzone"] {{
        background: {c.surface};
        border: 2px dashed {c.border};
        border-radius: 8px;
    }}

    QFrame[role="dropzone"][dragging="true"] {{
        border-color: {c.secondary};
        background: #17132A;
    }}

    QFrame[role="tile"] {{
        background: {c.surface};
        border: 1px solid {c.border};
        border-radius: 8px;
    }}

    QFrame[decision="keep"] {{
        border-color: {c.accent};
    }}

    QFrame[decision="review"] {{
        border-color: {c.warning};
    }}

    QFrame[decision="discard"] {{
        border-color: {c.destructive};
    }}

    QLabel[role="title"] {{
        color: {c.foreground};
        font-size: {t.xl}px;
        font-weight: 700;
    }}

    QLabel[role="section"] {{
        color: {c.foreground};
        font-size: {t.lg}px;
        font-weight: 700;
    }}

    QLabel[role="muted"] {{
        color: {c.muted_foreground};
        font-size: {t.sm}px;
    }}

    QLabel[role="metric"] {{
        color: {c.primary};
        font-size: {t.lg}px;
        font-weight: 700;
    }}

    QPushButton {{
        min-height: 44px;
        padding: 0 16px;
        border: 1px solid transparent;
        border-radius: 8px;
        background: {c.primary};
        color: {c.on_primary};
        font-weight: 600;
    }}

    QPushButton:hover {{
        background: #7DD3FC;
    }}

    QPushButton:pressed {{
        background: #0EA5E9;
    }}

    QPushButton:disabled {{
        background: {c.muted};
        color: {c.muted_foreground};
    }}

    QPushButton[variant="secondary"] {{
        background: {c.muted};
        color: {c.foreground};
        border-color: {c.border};
    }}

    QPushButton[variant="secondary"]:hover {{
        border-color: {c.primary};
        background: #2A3038;
    }}

    QPushButton[variant="accent"] {{
        background: {c.accent};
    }}

    QPushButton[variant="accent"]:hover {{
        background: #047857;
    }}

    QLineEdit, QComboBox {{
        min-height: 40px;
        border: 1px solid {c.border};
        border-radius: 8px;
        background: {c.surface};
        color: {c.foreground};
        padding: 0 10px;
    }}

    QLineEdit:focus, QComboBox:focus, QPushButton:focus, QSlider:focus {{
        border: 2px solid {c.ring};
    }}

    QComboBox QAbstractItemView {{
        background: {c.surface};
        color: {c.foreground};
        border: 1px solid {c.border};
        selection-background-color: {c.muted};
    }}

    QSlider::groove:horizontal {{
        height: 6px;
        background: {c.border};
        border-radius: 3px;
    }}

    QSlider::sub-page:horizontal {{
        background: {c.secondary};
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        width: 20px;
        height: 20px;
        margin: -8px 0;
        background: {c.surface};
        border: 2px solid {c.secondary};
        border-radius: 10px;
    }}

    QProgressBar {{
        min-height: 8px;
        max-height: 8px;
        border: 0;
        border-radius: 4px;
        background: {c.muted};
    }}

    QProgressBar::chunk {{
        border-radius: 4px;
        background: {c.accent};
    }}

    QTableWidget {{
        background: {c.surface};
        alternate-background-color: #111318;
        border: 1px solid {c.border};
        border-radius: 8px;
        gridline-color: {c.border};
        selection-background-color: #134E4A;
        selection-color: {c.foreground};
    }}

    QHeaderView::section {{
        background: {c.muted};
        color: {c.foreground};
        border: 0;
        border-bottom: 1px solid {c.border};
        padding: 8px;
        font-weight: 600;
    }}

    QScrollArea#photoGrid {{
        background: transparent;
        border: 0;
    }}

    QWidget#masonryContainer {{
        background: transparent;
    }}

    QLabel[role="photo-preview"] {{
        background: transparent;
    }}

    QLabel[role="photo-caption"] {{
        background: transparent;
        color: {c.muted_foreground};
        font-size: {t.sm}px;
    }}

    QFrame[role="photo-card"] {{
        background: {c.surface};
        border: 1px solid {c.border};
        border-radius: 8px;
        color: {c.foreground};
    }}

    QFrame[role="photo-card"]:hover {{
        border-color: {c.primary};
        background: #1A1D23;
    }}

    QScrollBar:vertical {{
        width: 10px;
        background: transparent;
    }}

    QScrollBar::handle:vertical {{
        min-height: 32px;
        border-radius: 5px;
        background: {c.border};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    """
