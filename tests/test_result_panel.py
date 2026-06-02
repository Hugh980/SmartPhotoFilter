from __future__ import annotations

from PyQt6.QtWidgets import QAbstractItemView, QApplication

from src.gui.widgets.result_panel import ResultPanel


def test_result_table_is_read_only():
    app = QApplication.instance() or QApplication([])
    panel = ResultPanel()
    table = panel.findChild(QAbstractItemView)

    assert table is not None
    assert table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers

    panel.close()
    app.processEvents()
