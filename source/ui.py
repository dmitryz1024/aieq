from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QCursor,
    QDesktopServices,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QRegion,
    QTextDocument,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDial,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStyleOptionViewItem,
    QStylePainter,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .ai import AiEqualizerService, AiPresetResult, list_local_models, read_gguf_context_length
from .audio import AudioDevice, AudioEngine, AudioStreamSetting, list_audio_devices, list_supported_stream_settings, refresh_audio_backend
from .autoeq_service import AutoEqPresetResult, build_autoeq_preset_result
from .chat_storage import ChatSession, ChatStore, chat_title_from_first_user_message
from .curves import DEVICE_CURVES_DIR, TARGET_CURVES_DIR, FrequencyCurve, ensure_curve_dirs, list_curves
from .dsp import DEFAULT_SAMPLE_RATE, GRAPH_FREQS, preset_response_db
from .models import FILTER_TYPES, EqFilter, Preset, flat_preset
from .storage import PresetStore

CURVE_COLORS = [
    "#f2b84b",
    "#5aa9ff",
    "#d984ff",
    "#ff8a4c",
    "#7ddc63",
    "#e85d75",
    "#9ad7ff",
    "#c792ea",
    "#ffcb6b",
    "#82aaff",
    "#f78c6c",
    "#a6e22e",
    "#c3e88d",
    "#ff6f91",
]
CURRENT_COLOR = "#05e5b6"
TARGET_COLOR = "#D44444"
USER_CHAT_COLOR = "#D44444"
DEVICE_CURVE_COLOR = "#8b9098"
NEW_PRESET_ID = "__new__"
DEFAULT_WINDOW_WIDTH = 1355
DEFAULT_WINDOW_HEIGHT = 712
DEFAULT_SPLITTER_SIZES = (925, 430)
LEGEND_LABEL_MAX_CHARS = 34
MAX_COMPARE_PRESETS = 15
FILTER_LABEL_FONT_PX = 13
TITLE_BAR_HEIGHT = 38
TITLE_CONTROLS_WIDTH = 104
WINDOW_RADIUS = 10
DEFAULT_LANGUAGE_CODE = "ru"
LANGUAGES_DIR = "languages"
SETTINGS_PANEL_WIDTH = 411
SETTINGS_AI_LABEL_WIDTH = 145
SETTINGS_AI_HINT_WIDTH = 88
POPUP_OPTION_DEFAULT_ROW_HEIGHT = 28
POPUP_OPTION_HOVER_HEIGHT = 28
POPUP_OPTION_TEXT_PADDING_X = 8
POPUP_LIST_VIEWPORT_PADDING_X = 12
POPUP_LIST_CHROME_HEIGHT = 12
FILTER_TYPE_ICON_FILES = {
    "peaking": "peaking.png",
    "low_shelf": "low shelf.png",
    "high_shelf": "high shelf.png",
    "low_pass": "low pass.png",
    "high_pass": "high pass.png",
    "band_pass": "band pass.png",
    "notch": "notch.png",
}
TOOLTIP_GAP = 4


def popup_list_height_for_rows(_list_widget: QListWidget, rows: int, row_height: int = POPUP_OPTION_DEFAULT_ROW_HEIGHT) -> int:
    rows = max(1, rows)
    return rows * row_height + POPUP_LIST_CHROME_HEIGHT


def set_popup_list_rows(list_widget: QListWidget, rows: int, row_height: int = POPUP_OPTION_DEFAULT_ROW_HEIGHT) -> None:
    list_widget.setSpacing(0)
    list_widget.setAutoScroll(False)
    list_widget.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
    list_widget.setProperty("optionRowHeight", row_height)
    list_widget.doItemsLayout()
    list_widget.setFixedHeight(popup_list_height_for_rows(list_widget, rows, row_height))


def load_language_bundle(code: str) -> tuple[str, str, dict[str, str]]:
    path = resource_path(f"{LANGUAGES_DIR}/{code}.json")
    if not path.exists() and code != DEFAULT_LANGUAGE_CODE:
        path = resource_path(f"{LANGUAGES_DIR}/{DEFAULT_LANGUAGE_CODE}.json")
    if not path.exists():
        return DEFAULT_LANGUAGE_CODE, "Русский", {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_LANGUAGE_CODE, "Русский", {}
    meta = data.get("meta", {})
    strings = data.get("strings", {})
    language_code = str(meta.get("code") or code or DEFAULT_LANGUAGE_CODE)
    language_name = str(meta.get("name") or language_code)
    if not isinstance(strings, dict):
        strings = {}
    return language_code, language_name, {str(key): str(value) for key, value in strings.items()}


def list_language_options() -> list[tuple[str, str]]:
    root = resource_path(LANGUAGES_DIR)
    options: list[tuple[str, str]] = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            meta = data.get("meta", {})
            code = str(meta.get("code") or path.stem)
            name = str(meta.get("name") or code)
            options.append((code, name))
    if not options:
        options.append((DEFAULT_LANGUAGE_CODE, "Русский"))
    return options


def elide_middle(text: str, max_chars: int = LEGEND_LABEL_MAX_CHARS) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    if max_chars <= 3:
        return "." * max_chars
    left = (max_chars - 3 + 1) // 2
    right = max_chars - 3 - left
    return f"{clean[:left]}...{clean[-right:]}"


def popup_text_width(widget: QWidget) -> int:
    return max(1, widget.width() - POPUP_LIST_VIEWPORT_PADDING_X - 2 * POPUP_OPTION_TEXT_PADDING_X)


def tooltip_for_elided_text(text: str, widget: QWidget, *, display_text: str | None = None) -> str:
    shown = display_text if display_text is not None else text
    if " ".join(text.split()) != " ".join(shown.split()):
        return text
    metrics = widget.fontMetrics()
    elided = metrics.elidedText(shown, Qt.TextElideMode.ElideRight, popup_text_width(widget))
    return text if elided != shown else ""


def wheel_steps_from_event(event) -> int:
    delta = event.angleDelta().y()
    if delta == 0:
        delta = event.pixelDelta().y()
    if delta == 0:
        return 0
    return int(delta / 120) if abs(delta) >= 120 else (1 if delta > 0 else -1)


def resource_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    candidates = [
        Path(getattr(sys, "_MEIPASS")) / relative if hasattr(sys, "_MEIPASS") else None,
        Path.cwd() / relative,
        Path(__file__).resolve().parent.parent / relative,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return Path(__file__).resolve().parent.parent / relative


def resource_asset_path(asset_name: str) -> Path:
    path = Path(asset_name)
    primary = resource_path(f"assets/{path.name}")
    if primary.exists():
        return primary
    if path.suffix.lower() == ".png":
        svg = resource_path(f"assets/{path.with_suffix('.svg').name}")
        if svg.exists():
            return svg
    if path.suffix.lower() == ".svg":
        png = resource_path(f"assets/{path.with_suffix('.png').name}")
        if png.exists():
            return png
    return primary


def current_device_pixel_ratio() -> float:
    app = QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return 1.0
    return max(1.0, float(screen.devicePixelRatio()))


def load_asset_pixmap(asset_name: str, size: QSize) -> QPixmap:
    path = resource_asset_path(asset_name)
    dpr = current_device_pixel_ratio()
    pixel_size = QSize(
        max(1, int(round(size.width() * dpr))),
        max(1, int(round(size.height() * dpr))),
    )
    if path.suffix.lower() == ".svg":
        renderer = QSvgRenderer(str(path))
        if renderer.isValid():
            pixmap = QPixmap(pixel_size)
            pixmap.fill(Qt.GlobalColor.transparent)
            pixmap.setDevicePixelRatio(dpr)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            renderer.render(painter, QRectF(0, 0, size.width(), size.height()))
            painter.end()
            return pixmap
    pixmap = QPixmap(str(path))
    if not pixmap.isNull():
        pixmap = pixmap.scaled(
            pixel_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        pixmap.setDevicePixelRatio(dpr)
    return pixmap


class LabelPaddedAxisItem(pg.AxisItem):
    def __init__(self, *args, label_offset: tuple[float, float] = (0.0, 0.0), **kwargs) -> None:
        self._label_offset = QPointF(*label_offset)
        super().__init__(*args, **kwargs)

    def setLabel(self, *args, **kwargs):  # type: ignore[override]
        result = super().setLabel(*args, **kwargs)
        QTimer.singleShot(0, self._recenter_label)
        return result

    def resizeEvent(self, ev=None):  # type: ignore[override]
        super().resizeEvent(ev)
        self._recenter_label(apply_perpendicular_offset=True)

    def _recenter_label(self, *, apply_perpendicular_offset: bool = False) -> None:
        if self.label is None:
            return
        rect = self.label.boundingRect()
        pos = self.label.pos()
        if self.orientation in {"bottom", "top"}:
            pos.setX((self.width() - rect.width()) / 2.0 + self._label_offset.x())
            if apply_perpendicular_offset:
                pos.setY(pos.y() + self._label_offset.y())
        else:
            pos.setY((self.height() + rect.width()) / 2.0 + self._label_offset.y())
            if apply_perpendicular_offset:
                pos.setX(pos.x() + self._label_offset.x())
        self.label.setPos(pos)


class FrequencyAxisItem(LabelPaddedAxisItem):
    FREQUENCY_TICKS: tuple[tuple[float, str], ...] = (
        (20.0, "20"),
        (50.0, "50"),
        (100.0, "100"),
        (200.0, "200"),
        (500.0, "500"),
        (1000.0, "1k"),
        (2000.0, "2k"),
        (5000.0, "5k"),
        (10000.0, "10k"),
        (20000.0, "20k"),
    )

    def tickValues(self, minVal, maxVal, size):  # type: ignore[override]
        if size < 560:
            selected = self.FREQUENCY_TICKS[::2]
        else:
            selected = self.FREQUENCY_TICKS
        values = [np.log10(freq) for freq, _label in selected if minVal <= np.log10(freq) <= maxVal]
        return [(1.0, values)]

    def tickStrings(self, values, scale, spacing):  # type: ignore[override]
        labels: list[str] = []
        for value in values:
            nearest = min(self.FREQUENCY_TICKS, key=lambda item: abs(np.log10(item[0]) - value))
            if abs(np.log10(nearest[0]) - value) < 1e-6:
                labels.append(nearest[1])
            else:
                labels.append("")
        return labels


class RoundedPlotWidget(pg.PlotWidget):
    RADIUS = 7.0

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.update_rounded_mask()

    def update_rounded_mask(self) -> None:
        rect = QRectF(self.rect())
        if rect.isEmpty():
            return
        path = QPainterPath()
        path.addRoundedRect(rect, self.RADIUS, self.RADIUS)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))


class RoundedPopupPanel(QWidget):
    RADIUS = 8.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("searchPopupPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlags(
            Qt.WindowType.Popup
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.update_rounded_mask()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.update_rounded_mask()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor("#313640"), 1.0))
        painter.setBrush(QColor("#181a1f"))
        painter.drawRoundedRect(rect, self.RADIUS, self.RADIUS)

    def update_rounded_mask(self) -> None:
        rect = QRectF(self.rect())
        if rect.isEmpty():
            return
        path = QPainterPath()
        path.addRoundedRect(rect, self.RADIUS, self.RADIUS)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))


class HoverListWidget(QListWidget):
    hovered_row_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hovered_row = -1
        self._hover_tracked_widgets: set[QObject] = set()
        self._wheel_enabled = True
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def set_wheel_enabled(self, enabled: bool) -> None:
        self._wheel_enabled = enabled

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not self._wheel_enabled:
            event.ignore()
            return
        super().wheelEvent(event)

    def track_hover_widget(self, widget: QWidget) -> None:
        widget.setMouseTracking(True)
        widget.installEventFilter(self)
        self._hover_tracked_widgets.add(widget)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        super().mouseMoveEvent(event)
        index = self.indexAt(event.pos())
        self._set_hovered_row(index.row() if index.isValid() else -1)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        self._set_hovered_row(-1)

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if obj in self._hover_tracked_widgets:
            if event.type() == QEvent.Type.MouseMove:
                global_position = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else obj.mapToGlobal(event.pos())
                index = self.indexAt(self.viewport().mapFromGlobal(global_position))
                self._set_hovered_row(index.row() if index.isValid() else -1)
            elif event.type() == QEvent.Type.Leave:
                index = self.indexAt(self.viewport().mapFromGlobal(QCursor.pos()))
                self._set_hovered_row(index.row() if index.isValid() else -1)
        return super().eventFilter(obj, event)

    def _set_hovered_row(self, row: int) -> None:
        if row == self._hovered_row:
            return
        self._hovered_row = row
        self.hovered_row_changed.emit(row)

    def row_at_global_pos(self, pos: QPoint) -> int:
        index = self.indexAt(self.viewport().mapFromGlobal(pos))
        return index.row() if index.isValid() else -1


class PopupOptionDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index) -> QSize:  # type: ignore[override]
        parent = self.parent()
        row_height = POPUP_OPTION_DEFAULT_ROW_HEIGHT
        if isinstance(parent, QWidget):
            value = parent.property("optionRowHeight")
            if value:
                row_height = int(value)
        return QSize(1, row_height)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # type: ignore[override]
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        enabled = bool(opt.state & QStyle.StateFlag.State_Enabled)
        hovered = enabled and bool(opt.state & QStyle.StateFlag.State_MouseOver)
        painter.save()
        painter.setClipRect(opt.rect)
        painter.fillRect(opt.rect, QColor("#111318"))
        if hovered:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2d3440"))
            hover_height = min(POPUP_OPTION_HOVER_HEIGHT, opt.rect.height())
            y = opt.rect.y() + (opt.rect.height() - hover_height) / 2.0
            rect = QRectF(opt.rect.x(), y, opt.rect.width(), hover_height)
            painter.drawRoundedRect(rect, 4.0, 4.0)
        painter.setFont(opt.font)
        painter.setPen(QColor("#cfd6df" if enabled else "#69727f"))
        text_rect = opt.rect.adjusted(POPUP_OPTION_TEXT_PADDING_X, 0, -POPUP_OPTION_TEXT_PADDING_X, 0)
        if not opt.icon.isNull():
            icon_size = opt.decorationSize if opt.decorationSize.isValid() else QSize(16, 16)
            icon_rect = QRect(
                text_rect.x(),
                text_rect.y() + max(0, (text_rect.height() - icon_size.height()) // 2),
                icon_size.width(),
                icon_size.height(),
            )
            opt.icon.paint(painter, icon_rect, Qt.AlignmentFlag.AlignCenter)
            text_rect.setLeft(icon_rect.right() + 6)
        text = painter.fontMetrics().elidedText(
            opt.text,
            Qt.TextElideMode.ElideRight,
            max(0, text_rect.width()),
        )
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        painter.restore()


class TooltipTextLabel(QWidget):
    def __init__(self, text: str, text_width: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("filterTypeInfoText")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._text = ""
        self._text_width = max(1, int(text_width))
        self._document = QTextDocument(self)
        self._document.setDocumentMargin(0)
        self.set_text(text, self._text_width)

    def set_text(self, text: str, text_width: int | None = None) -> None:
        self._text = text
        if text_width is not None:
            self._text_width = max(1, int(text_width))
        self._rebuild_document()

    def _rebuild_document(self) -> None:
        safe_text = html.escape(self._text).replace("\n", "<br>")
        self._document.setDefaultFont(self.font())
        self._document.setDefaultStyleSheet(
            "body { margin: 0; padding: 0; } "
            "div { margin: 0; padding: 0; text-align: center; color: #cfd6df; }"
        )
        self._document.setHtml(f"<div>{safe_text}</div>")
        self._document.setTextWidth(self._text_width)
        height = max(1, int(np.ceil(self._document.size().height())) + 2)
        self.setMinimumWidth(self._text_width)
        self.setFixedHeight(height)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(self._text_width, self.height())

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self._document.drawContents(painter, QRectF(0, 0, self._text_width, self.height()))


class AieqLegendSample(pg.graphicsItems.LegendItem.ItemSample):
    def paint(self, painter, *args):  # type: ignore[override]
        if not self.item.isVisible():
            return
        opts = self.item.opts
        if opts.get("antialias"):
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(pg.mkPen(opts["pen"]))
        painter.drawLine(QPointF(0.0, 12.0), QPointF(20.0, 12.0))


class HoverLegendItem(pg.LegendItem):
    COLLAPSED_WIDTH = 24
    COLLAPSED_HEIGHT = 24
    COLLAPSED_ICON_SIZE = QSize(16, 16)
    MAX_ROWS_PER_COLUMN = 6
    EXPANDED_PADDING_X = 18
    EXPANDED_PADDING_Y = 14
    _collapsed_icon: QPixmap | None = None

    def __init__(self, *args, **kwargs) -> None:
        self._collapsed = False
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.set_collapsed(True)

    def addItem(self, item, name):  # type: ignore[override]
        full_name = str(name)
        display_name = elide_middle(full_name)
        super().addItem(item, display_name)
        if self.items:
            sample, label = self.items[-1]
            sample.setToolTip("")
            self._set_label_tooltip(label, full_name, display_name)
        self._reflow_items()
        self.set_collapsed(self._collapsed)

    def removeItem(self, item):  # type: ignore[override]
        super().removeItem(item)
        self._reflow_items()
        self.set_collapsed(self._collapsed)

    def setItemName(self, item, name: str) -> None:
        full_name = str(name)
        display_name = elide_middle(full_name)
        for sample, label in self.items:
            if getattr(sample, "item", None) is item:
                label.setText(display_name)
                sample.setToolTip("")
                self._set_label_tooltip(label, full_name, display_name)
                self._reflow_items()
                self.set_collapsed(self._collapsed)
                return

    def _set_label_tooltip(self, label, full_name: str, display_name: str) -> None:
        normalized = " ".join(full_name.split())
        tooltip = full_name if display_name != normalized else ""
        label.setToolTip("")
        text_item = getattr(label, "item", None)
        if text_item is not None:
            text_item.setToolTip(tooltip)

    def updateSize(self):  # type: ignore[override]
        if self._collapsed:
            self.setGeometry(0, 0, self.COLLAPSED_WIDTH, self.COLLAPSED_HEIGHT)
            return
        self._reflow_items()
        self._fit_expanded_geometry()

    def hoverEvent(self, ev):  # type: ignore[override]
        if ev.isExit():
            self.set_collapsed(True)
        else:
            self.set_collapsed(False)

    def mouseDragEvent(self, ev):  # type: ignore[override]
        ev.accept()
        self._reanchor()

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        for sample, label in self.items:
            sample.setVisible(not collapsed)
            label.setVisible(not collapsed)
        if collapsed:
            self.setMinimumSize(self.COLLAPSED_WIDTH, self.COLLAPSED_HEIGHT)
            self.setMaximumSize(self.COLLAPSED_WIDTH, self.COLLAPSED_HEIGHT)
            self.setGeometry(0, 0, self.COLLAPSED_WIDTH, self.COLLAPSED_HEIGHT)
        else:
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)
            self._reflow_items()
            self._fit_expanded_geometry()
        self._reanchor()
        self.update()

    def _reflow_items(self) -> None:
        for index in range(self.layout.count() - 1, -1, -1):
            self.layout.removeAt(index)
        columns = max(1, int(np.ceil(len(self.items) / self.MAX_ROWS_PER_COLUMN)))
        self.columnCount = columns
        self.rowCount = min(self.MAX_ROWS_PER_COLUMN, max(1, len(self.items)))
        for index, (sample, label) in enumerate(self.items):
            column = index // self.MAX_ROWS_PER_COLUMN
            row = index % self.MAX_ROWS_PER_COLUMN
            self.layout.addItem(sample, row, column * 2)
            self.layout.addItem(label, row, column * 2 + 1)

    def _fit_expanded_geometry(self) -> None:
        if self._collapsed:
            return
        width, height = self._expanded_size()
        self.setGeometry(
            0,
            0,
            width,
            height,
        )
        self._reanchor()

    def _expanded_size(self) -> tuple[float, float]:
        if not self.items:
            return float(self.COLLAPSED_WIDTH), float(self.COLLAPSED_HEIGHT)
        column_widths: list[float] = []
        column_heights: list[float] = []
        spacing_x = max(8.0, float(self.layout.horizontalSpacing()))
        spacing_y = max(4.0, float(self.layout.verticalSpacing()))
        for column_start in range(0, len(self.items), self.MAX_ROWS_PER_COLUMN):
            column_items = self.items[column_start : column_start + self.MAX_ROWS_PER_COLUMN]
            row_widths: list[float] = []
            row_heights: list[float] = []
            for sample, label in column_items:
                sample_width, sample_height, label_width, label_height = self._item_metrics(sample, label)
                row_widths.append(sample_width + spacing_x + label_width)
                row_heights.append(max(sample_height, label_height))
            column_widths.append(max(row_widths, default=0.0))
            column_heights.append(sum(row_heights) + spacing_y * max(0, len(row_heights) - 1))
        width = sum(column_widths) + spacing_x * max(0, len(column_widths) - 1) + self.EXPANDED_PADDING_X
        height = max(column_heights, default=0.0) + self.EXPANDED_PADDING_Y
        return max(width, float(self.COLLAPSED_WIDTH)), max(height, float(self.COLLAPSED_HEIGHT))

    def _item_metrics(self, sample, label) -> tuple[float, float, float, float]:
        sample_rect = sample.boundingRect()
        text_item = getattr(label, "item", None)
        label_rect = text_item.boundingRect() if text_item is not None else label.boundingRect()
        sample_width = max(float(sample_rect.width()), 20.0)
        sample_height = max(float(sample_rect.height()), 12.0)
        label_width = max(float(label_rect.width()), 1.0)
        label_height = max(float(label_rect.height()), 14.0)
        return sample_width, sample_height, label_width, label_height

    def _reanchor(self) -> None:
        if self.parentItem() is None:
            return
        offset = self.opts.get("offset")
        if offset is not None:
            self.setOffset(offset)

    @classmethod
    def _collapsed_icon_pixmap(cls) -> QPixmap:
        if cls._collapsed_icon is None:
            cls._collapsed_icon = load_asset_pixmap("legend.svg", cls.COLLAPSED_ICON_SIZE)
        return cls._collapsed_icon

    def paint(self, p, *args):  # type: ignore[override]
        rect = self.boundingRect().adjusted(0.5, 0.5, -0.5, -0.5)
        if self._collapsed:
            p.setPen(pg.mkPen("#59616e"))
            p.setBrush(pg.mkBrush(17, 19, 24, 230))
        else:
            p.setPen(pg.mkPen("#3a414d"))
            p.setBrush(pg.mkBrush(17, 19, 24, 210))
        p.drawRoundedRect(rect, 5.0, 5.0)
        if self._collapsed:
            icon = self._collapsed_icon_pixmap()
            if not icon.isNull():
                icon_size = icon.deviceIndependentSize()
                target = QRectF(
                    rect.center().x() - icon_size.width() / 2.0,
                    rect.center().y() - icon_size.height() / 2.0,
                    icon_size.width(),
                    icon_size.height(),
                )
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
                p.drawPixmap(target, icon, QRectF(icon.rect()))


class ElidingComboBox(QComboBox):
    def compactSizeHint(self) -> QSize:
        hint = super().sizeHint()
        if self.minimumWidth() > 0 and self.minimumWidth() == self.maximumWidth():
            hint.setWidth(self.minimumWidth())
            return hint
        min_chars = max(0, self.minimumContentsLength())
        if min_chars:
            text_width = self.fontMetrics().horizontalAdvance("M" * min_chars)
            hint.setWidth(text_width + 44)
        else:
            hint.setWidth(min(hint.width(), 220))
        return hint

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return self.compactSizeHint()

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return self.compactSizeHint()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        edit_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        text_width = max(0, edit_rect.width() - 2)
        option.currentText = self.fontMetrics().elidedText(
            option.currentText,
            Qt.TextElideMode.ElideRight,
            text_width,
        )
        painter = QStylePainter(self)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)

    def _available_popup_rect(self) -> QRect:
        window = self.window()
        if isinstance(window, QWidget) and window.isVisible():
            rect = QRect(window.mapToGlobal(QPoint(0, 0)), window.size())
        else:
            rect = QRect()
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            screen_rect = screen.availableGeometry()
            rect = rect.intersected(screen_rect) if not rect.isEmpty() else screen_rect
        return rect

    def _position_popup(self, container: QWidget, *, align_center: bool = False) -> None:
        available = self._available_popup_rect()
        x_offset = int((self.width() - container.width()) / 2) if align_center else 0
        anchor_below = self.mapToGlobal(QPoint(x_offset, self.height()))
        anchor_above = self.mapToGlobal(QPoint(x_offset, 0))
        if available.isEmpty():
            container.move(anchor_below)
            if hasattr(container, "update_rounded_mask"):
                container.update_rounded_mask()
            return

        x = max(available.left(), min(anchor_below.x(), available.right() - container.width() + 1))
        space_below = max(0, available.bottom() - anchor_below.y() + 1)
        space_above = max(0, anchor_above.y() - available.top())
        popup_height = container.height()
        if popup_height > space_below and space_above > space_below:
            y = max(available.top(), anchor_above.y() - popup_height)
        else:
            y = min(anchor_below.y(), available.bottom() - popup_height + 1)
        container.move(QPoint(x, max(available.top(), y)))
        if hasattr(container, "update_rounded_mask"):
            container.update_rounded_mask()


class CompactLabel(QLabel):
    def __init__(self, text: str = "", *, hint_width: int = 90) -> None:
        super().__init__(text)
        self._hint_width = hint_width

    def sizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().sizeHint()
        text_width = self.fontMetrics().horizontalAdvance(self.text()) + 2
        hint.setWidth(min(self._hint_width, max(1, text_width)))
        return hint

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().minimumSizeHint()
        text_width = self.fontMetrics().horizontalAdvance(self.text()) + 2
        hint.setWidth(min(self._hint_width, max(1, text_width)))
        return hint

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, max(0, self.width()))
        painter.drawText(self.rect(), self.alignment() | Qt.AlignmentFlag.AlignVCenter, text)


class CompactButton(QPushButton):
    def sizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().sizeHint()
        if self.minimumWidth() > 0 and self.minimumWidth() == self.maximumWidth():
            hint.setWidth(self.minimumWidth())
        return hint

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return self.sizeHint()


class SearchableComboBox(ElidingComboBox):
    MAX_RENDERED_RESULTS = 120

    def __init__(self, *, empty_text: str, search_text: str = "Поиск") -> None:
        super().__init__()
        self.empty_text = empty_text
        self.search_text = search_text
        self._search_popup: QWidget | None = None
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()

    def showPopup(self) -> None:  # type: ignore[override]
        if self._search_popup is not None:
            self._search_popup.close()
        container = RoundedPopupPanel(self.window())
        container.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        popup_width = max(80, self.width())
        container.setFixedWidth(popup_width)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        search = QLineEdit(container)
        search.setObjectName("comboSearch")
        search.setPlaceholderText(self.search_text)
        list_widget = HoverListWidget(container)
        list_widget.setObjectName("comboSearchList")
        list_widget.set_wheel_enabled(not bool(self.property("disablePopupWheel")))
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setSpacing(0)
        list_widget.setAutoScroll(False)
        list_widget.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        list_widget.setUniformItemSizes(True)
        list_widget.setItemDelegate(PopupOptionDelegate(list_widget))
        search.setFixedWidth(popup_width - 12)
        list_widget.setFixedWidth(popup_width - 12)
        list_widget.setMaximumHeight(260)
        layout.addWidget(search)
        layout.addWidget(list_widget)

        entries = [(index, self.itemText(index), self.itemText(index).casefold()) for index in range(self.count())]
        max_results = max(1, int(self.property("maxPopupResults") or self.MAX_RENDERED_RESULTS))
        preferred_rows = max(1, int(self.property("popupVisibleRows") or 8))

        def resize_popup() -> None:
            configured_row_height = self.property("popupRowHeight")
            row_height = int(configured_row_height) if configured_row_height else POPUP_OPTION_DEFAULT_ROW_HEIGHT
            list_widget.setProperty("optionRowHeight", row_height)
            list_widget.doItemsLayout()
            frame = list_widget.frameWidth() * 2
            visible_rows = min(max(list_widget.count(), 1), preferred_rows)
            list_widget.setFixedHeight(min(260, visible_rows * row_height + frame))
            margins = layout.contentsMargins()
            container.setFixedHeight(search.height() + layout.spacing() + list_widget.height() + margins.top() + margins.bottom())
            container.adjustSize()
            container.update_rounded_mask()
            container.move(self.mapToGlobal(self.rect().bottomLeft()))

        def populate(query: str = "") -> None:
            list_widget.setUpdatesEnabled(False)
            list_widget.clear()
            query = query.strip().casefold()
            shown = 0
            has_more = False
            for index, text, folded_text in entries:
                if query and query not in folded_text:
                    continue
                if shown >= max_results:
                    has_more = True
                    break
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, index)
                item.setToolTip(tooltip_for_elided_text(text, list_widget))
                if index == self.currentIndex():
                    item.setSelected(True)
                list_widget.addItem(item)
                shown += 1
            if shown == 0:
                item = QListWidgetItem(self.empty_text)
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                list_widget.addItem(item)
            elif has_more:
                item = QListWidgetItem("...")
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                list_widget.addItem(item)
            list_widget.setUpdatesEnabled(True)
            resize_popup()
            for row in range(list_widget.count()):
                item = list_widget.item(row)
                if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                    text = item.text()
                    item.setToolTip(tooltip_for_elided_text(text, list_widget))

        def choose(item: QListWidgetItem) -> None:
            index = item.data(Qt.ItemDataRole.UserRole)
            if index is None:
                return
            self.setCurrentIndex(int(index))
            container.close()

        filter_timer = QTimer(container)
        filter_timer.setSingleShot(True)
        pending_query = {"text": ""}

        def schedule_populate(query: str) -> None:
            pending_query["text"] = query
            filter_timer.start(35)

        filter_timer.timeout.connect(lambda: populate(pending_query["text"]))
        populate()
        search.textChanged.connect(schedule_populate)
        list_widget.itemClicked.connect(choose)

        self._search_popup = container
        container.destroyed.connect(lambda _obj=None: setattr(self, "_search_popup", None))
        QTimer.singleShot(0, search.setFocus)
        resize_popup()
        container.show()
        container.raise_()


class SimpleComboBox(ElidingComboBox):
    def __init__(self) -> None:
        super().__init__()
        self._simple_popup: QWidget | None = None
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if bool(self.property("enableCollapsedWheel")) and self.isEnabled() and self.count() > 1:
            steps = wheel_steps_from_event(event)
            if steps:
                next_index = max(0, min(self.count() - 1, self.currentIndex() - steps))
                if next_index != self.currentIndex():
                    self.setCurrentIndex(next_index)
                event.accept()
                return
        event.ignore()

    def showPopup(self) -> None:  # type: ignore[override]
        if self._simple_popup is not None:
            self._simple_popup.close()
        container = RoundedPopupPanel(self.window())
        container.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        popup_width = max(80, self.width())
        container.setFixedWidth(popup_width)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)

        list_widget = HoverListWidget(container)
        list_widget.setObjectName("comboSearchList")
        list_widget.set_wheel_enabled(not bool(self.property("disablePopupWheel")))
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setUniformItemSizes(True)
        list_widget.setItemDelegate(PopupOptionDelegate(list_widget))
        list_widget.setFixedWidth(popup_width - 12)
        list_widget.setMaximumHeight(260)
        list_widget.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        layout.addWidget(list_widget)

        for index in range(self.count()):
            item = QListWidgetItem(self.itemText(index))
            item.setData(Qt.ItemDataRole.UserRole, index)
            if not (self.model().flags(self.model().index(index, self.modelColumn())) & Qt.ItemFlag.ItemIsEnabled):
                item.setFlags(Qt.ItemFlag.NoItemFlags)
            list_widget.addItem(item)
        if self.count() == 0:
            item = QListWidgetItem("Нет вариантов")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            list_widget.addItem(item)

        configured_row_height = self.property("popupRowHeight")
        row_height = int(configured_row_height) if configured_row_height else POPUP_OPTION_DEFAULT_ROW_HEIGHT
        preferred_rows = int(self.property("popupVisibleRows") or 8)
        visible_rows = min(max(list_widget.count(), 1), max(1, preferred_rows))
        set_popup_list_rows(list_widget, visible_rows, row_height)
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setToolTip(tooltip_for_elided_text(item.text(), list_widget))
        margins = layout.contentsMargins()
        container.setFixedHeight(list_widget.height() + margins.top() + margins.bottom())

        def choose(item: QListWidgetItem) -> None:
            index = item.data(Qt.ItemDataRole.UserRole)
            if index is None:
                return
            self.setCurrentIndex(int(index))
            container.close()

        list_widget.itemClicked.connect(choose)
        self._simple_popup = container
        container.destroyed.connect(lambda _obj=None: setattr(self, "_simple_popup", None))
        container.adjustSize()
        container.update_rounded_mask()
        if bool(self.property("constrainPopupToWindow")):
            self._position_popup(container, align_center=bool(self.property("alignPopupCenter")))
        elif bool(self.property("alignPopupCenter")):
            x = int((self.width() - container.width()) / 2)
            container.move(self.mapToGlobal(QPoint(x, self.height())))
        else:
            container.move(self.mapToGlobal(self.rect().bottomLeft()))
        container.show()
        container.raise_()


class CleanDoubleSpinBox(QDoubleSpinBox):
    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        QTimer.singleShot(0, self.clear_related_selections)

    def focusOutEvent(self, event) -> None:  # type: ignore[override]
        self.clear_selection()
        super().focusOutEvent(event)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        self.clear_related_selections()
        if not self.isEnabled():
            event.ignore()
            return
        steps = wheel_steps_from_event(event)
        if steps == 0:
            event.ignore()
            return
        self.stepBy(steps)
        event.accept()
        self.clear_related_selections()
        QTimer.singleShot(0, self.clear_related_selections)
        QTimer.singleShot(16, self.clear_related_selections)

    def clear_selection(self) -> None:
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.deselect()

    def clear_edit_focus(self) -> None:
        self.clear_selection()
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.clearFocus()
        self.clearFocus()

    def clear_related_selections(self) -> None:
        parent = self.parentWidget()
        while parent is not None:
            editors = getattr(parent, "value_editors", None)
            if editors is not None:
                for editor in editors:
                    if hasattr(editor, "clear_selection"):
                        editor.clear_selection()
                return
            parent = parent.parentWidget()
        self.clear_selection()


class NoWheelDoubleSpinBox(CleanDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()


class AieqCheckBox(QCheckBox):
    INDICATOR_SIZE = 14
    TEXT_GAP = 8
    LEFT_PADDING = 0

    def enterEvent(self, event) -> None:  # type: ignore[override]
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        enabled = self.isEnabled()
        if bool(self.property("searchMenuCheckbox")) and bool(self.property("searchMenuHovered")):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2d3440"))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.0, -0.5, 0.0), 4.0, 4.0)
        size = self.INDICATOR_SIZE
        has_text = bool(self.text())
        text_gap = float(self.property("textGap") or self.TEXT_GAP)
        x = 0.0 if has_text else max(0.0, (self.width() - size) / 2.0)
        if has_text and bool(self.property("centerContent")):
            metrics = QFontMetricsF(painter.font())
            total_width = size + text_gap + metrics.horizontalAdvance(self.text())
            x = max(0.0, (self.width() - total_width) / 2.0)
        elif has_text:
            x = float(self.property("leftPadding") or self.LEFT_PADDING)
        y = max(0.0, (self.height() - size) / 2.0)
        rect = QRectF(x + 0.5, y + 0.5, size - 1.0, size - 1.0)

        menu_hover = bool(self.property("searchMenuHovered"))
        border_color = "#111318" if enabled and menu_hover else ("#313640" if enabled else "#252b35")
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(border_color), 1.0))
        painter.drawRoundedRect(rect, 3.0, 3.0)

        if self.isChecked():
            path = QPainterPath()
            path.moveTo(x + 3.4, y + 7.2)
            path.lineTo(x + 6.0, y + 9.7)
            path.lineTo(x + 10.8, y + 4.4)
            pen = QPen(QColor(CURRENT_COLOR if enabled else "#2f7d6b"), 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)
        if has_text:
            painter.setPen(QColor("#cfd6df" if enabled else "#69727f"))
            text_rect = QRectF(size + text_gap, 0, max(0, self.width() - size - text_gap), self.height())
            if bool(self.property("centerContent")):
                text_rect = QRectF(x + size + text_gap, 0, max(0, self.width() - x - size - text_gap), self.height())
            elif has_text:
                text_rect = QRectF(x + size + text_gap, 0, max(0, self.width() - x - size - text_gap), self.height())
            text = painter.fontMetrics().elidedText(
                self.text(),
                Qt.TextElideMode.ElideRight,
                max(0, int(text_rect.width())),
            )
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)


class WindowControlButton(QPushButton):
    SCALE = 0.72

    def __init__(self, symbol: str, *, close_button: bool = False) -> None:
        super().__init__()
        self._symbol = symbol
        self._close_button = close_button
        self.setFixedSize(30, 24)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_symbol(self, symbol: str) -> None:
        self._symbol = symbol
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        hovered = self.underMouse()
        pressed = self.isDown()
        if hovered:
            if self._close_button:
                fill = QColor("#6a2930" if pressed else "#532126")
                pen = QPen(QColor("#d44444"), 1.0)
            else:
                fill = QColor("#1f242c" if pressed else "#242933")
                pen = QPen(QColor("#3a414d"), 1.0)
            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6.0, 6.0)

        color = QColor("#ffffff" if hovered else "#cfd6df")
        pen = QPen(color, 1.7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        center = QRectF(self.rect()).center()
        scale = self.SCALE
        if self._symbol == "\u2212":
            painter.drawLine(QPointF(center.x() - 5.0 * scale, center.y()), QPointF(center.x() + 5.0 * scale, center.y()))
        elif self._symbol == "\u25a1":
            size = 10.0 * scale
            painter.drawRect(QRectF(center.x() - size / 2.0, center.y() - size / 2.0, size, size))
        elif self._symbol == "\u2750":
            size = 9.0 * scale
            offset = 3.0 * scale
            painter.drawRect(QRectF(center.x() - size / 2.0 + offset / 2.0, center.y() - size / 2.0 - offset / 2.0, size, size))
            painter.drawRect(QRectF(center.x() - size / 2.0 - offset / 2.0, center.y() - size / 2.0 + offset / 2.0, size, size))
        else:
            extent = 4.5 * scale
            painter.drawLine(QPointF(center.x() - extent, center.y() - extent), QPointF(center.x() + extent, center.y() + extent))
            painter.drawLine(QPointF(center.x() + extent, center.y() - extent), QPointF(center.x() - extent, center.y() + extent))


def _dialog_text(parent: QWidget | None, key: str, fallback: str) -> str:
    translator = getattr(parent, "t", None)
    if callable(translator):
        translated = translator(key)
        if translated != key:
            return translated
    return fallback


def _standard_button_text(parent: QWidget | None, button: QMessageBox.StandardButton) -> str:
    labels = {
        QMessageBox.StandardButton.Ok: ("dialog.ok", "OK"),
        QMessageBox.StandardButton.Yes: ("dialog.yes", "Yes"),
        QMessageBox.StandardButton.No: ("dialog.no", "No"),
        QMessageBox.StandardButton.Save: ("dialog.save", "Save"),
        QMessageBox.StandardButton.Discard: ("dialog.discard", "Discard"),
        QMessageBox.StandardButton.Cancel: ("dialog.cancel", "Cancel"),
    }
    key, fallback = labels.get(button, ("dialog.ok", "OK"))
    return _dialog_text(parent, key, fallback)


def _ordered_standard_buttons(buttons: QMessageBox.StandardButtons) -> list[QMessageBox.StandardButton]:
    order = [
        QMessageBox.StandardButton.Save,
        QMessageBox.StandardButton.Yes,
        QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Discard,
        QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Ok,
    ]
    return [button for button in order if button & buttons]


class AieqDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        *,
        title: str,
        message: str,
        buttons: QMessageBox.StandardButtons,
        default_button: QMessageBox.StandardButton,
        input_label: str | None = None,
        input_text: str = "",
    ) -> None:
        super().__init__(parent)
        self.selected_button = QMessageBox.StandardButton.Cancel
        self._title_drag_pos: QPoint | None = None
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumWidth(360)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.dialog_frame = QFrame()
        self.dialog_frame.setObjectName("dialogFrame")
        frame_layout = QVBoxLayout(self.dialog_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        outer.addWidget(self.dialog_frame)

        title_bar = QWidget()
        title_bar.setObjectName("windowTitleBar")
        title_bar.setFixedHeight(34)
        title_bar.installEventFilter(self)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 5, 8, 5)
        title_layout.setSpacing(6)
        title_label = QLabel(title)
        title_label.setObjectName("dialogTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        title_layout.addWidget(title_label, 1)
        close_button = WindowControlButton("\u00d7", close_button=True)
        close_button.setFixedSize(26, 22)
        close_button.clicked.connect(self.reject)
        title_layout.addWidget(close_button, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        frame_layout.addWidget(title_bar)

        body = QWidget()
        body.setObjectName("dialogBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 14, 16, 14)
        body_layout.setSpacing(10)
        if message:
            message_label = QLabel(message)
            message_label.setObjectName("dialogMessage")
            message_label.setWordWrap(True)
            message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            body_layout.addWidget(message_label)

        self.input_edit: QLineEdit | None = None
        if input_label is not None:
            label = QLabel(input_label)
            label.setObjectName("dialogInputLabel")
            body_layout.addWidget(label)
            self.input_edit = QLineEdit()
            self.input_edit.setObjectName("dialogInputEdit")
            self.input_edit.setText(input_text)
            self.input_edit.selectAll()
            body_layout.addWidget(self.input_edit)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 4, 0, 0)
        button_row.setSpacing(8)
        button_row.addStretch(1)
        for standard_button in _ordered_standard_buttons(buttons):
            button = QPushButton(_standard_button_text(parent, standard_button))
            button.setFixedHeight(30)
            if standard_button == default_button:
                button.setDefault(True)
                button.setProperty("defaultDialogButton", True)
            button.clicked.connect(lambda _checked=False, value=standard_button: self.finish(value))
            button_row.addWidget(button)
        body_layout.addLayout(button_row)
        frame_layout.addWidget(body)

    def finish(self, button: QMessageBox.StandardButton) -> None:
        self.selected_button = button
        self.accept()

    def reject(self) -> None:  # type: ignore[override]
        self.selected_button = QMessageBox.StandardButton.Cancel
        super().reject()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        rect = QRectF(self.rect())
        if rect.isEmpty():
            return
        path = QPainterPath()
        path.addRoundedRect(rect, WINDOW_RADIUS, WINDOW_RADIUS)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self.input_edit is not None:
            self.input_edit.setFocus(Qt.FocusReason.PopupFocusReason)

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._title_drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            return True
        if event.type() == QEvent.Type.MouseMove and self._title_drag_pos is not None:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.move(event.globalPosition().toPoint() - self._title_drag_pos)
                return True
        if event.type() == QEvent.Type.MouseButtonRelease:
            self._title_drag_pos = None
            return True
        return super().eventFilter(obj, event)


def ask_custom_question(
    parent: QWidget,
    title: str,
    message: str,
    buttons: QMessageBox.StandardButtons,
    default_button: QMessageBox.StandardButton,
) -> QMessageBox.StandardButton:
    dialog = AieqDialog(
        parent,
        title=title,
        message=message,
        buttons=buttons,
        default_button=default_button,
    )
    dialog.exec()
    if dialog.selected_button == QMessageBox.StandardButton.Cancel and not (buttons & QMessageBox.StandardButton.Cancel):
        if buttons & QMessageBox.StandardButton.No:
            return QMessageBox.StandardButton.No
    return dialog.selected_button


def show_custom_message(parent: QWidget, title: str, message: str) -> None:
    dialog = AieqDialog(
        parent,
        title=title,
        message=message,
        buttons=QMessageBox.StandardButton.Ok,
        default_button=QMessageBox.StandardButton.Ok,
    )
    dialog.exec()


def get_custom_text(parent: QWidget, title: str, label: str, *, text: str = "") -> tuple[str, bool]:
    dialog = AieqDialog(
        parent,
        title=title,
        message="",
        input_label=label,
        input_text=text,
        buttons=QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        default_button=QMessageBox.StandardButton.Ok,
    )
    accepted = dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_button == QMessageBox.StandardButton.Ok
    return (dialog.input_edit.text() if dialog.input_edit is not None else "", accepted)


class AudioTransportIcon(QLabel):
    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ffffff"))
        rect = QRectF(self.rect())
        center = rect.center()
        if bool(self.property("running")):
            size = 8.0
            painter.drawRect(QRectF(center.x() - size / 2.0, center.y() - size / 2.0, size, size))
            return
        path = QPainterPath()
        path.moveTo(center.x() - 2.8, center.y() - 4.8)
        path.lineTo(center.x() - 2.8, center.y() + 4.8)
        path.lineTo(center.x() + 5.0, center.y())
        path.closeSubpath()
        painter.drawPath(path)


class RefreshIcon(QLabel):
    ICON_SIZE = QSize(16, 16)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = load_asset_pixmap("sync.svg", self.ICON_SIZE)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        enabled = self.isEnabled() and (self.parentWidget() is None or self.parentWidget().isEnabled())
        if self._pixmap.isNull():
            return
        target = QRectF(
            (self.width() - self.ICON_SIZE.width()) / 2.0,
            (self.height() - self.ICON_SIZE.height()) / 2.0,
            self.ICON_SIZE.width(),
            self.ICON_SIZE.height(),
        )
        painter.setOpacity(1.0 if enabled else 0.45)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))


class AssetIcon(QLabel):
    def __init__(
        self,
        asset_name: str,
        size: QSize = QSize(16, 16),
        parent: QWidget | None = None,
        offset: QPoint | None = None,
    ) -> None:
        super().__init__(parent)
        self._size = QSize(size)
        self._offset = QPoint(offset) if offset is not None else QPoint(0, 0)
        self._pixmap = load_asset_pixmap(asset_name, self._size)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        enabled = self.isEnabled() and (self.parentWidget() is None or self.parentWidget().isEnabled())
        if self._pixmap.isNull():
            return
        target = QRectF(
            (self.width() - self._size.width()) / 2.0 + self._offset.x(),
            (self.height() - self._size.height()) / 2.0 + self._offset.y(),
            self._size.width(),
            self._size.height(),
        )
        painter.setOpacity(1.0 if enabled else 0.45)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))


class InlineAssetButton(QPushButton):
    def __init__(
        self,
        asset_name: str,
        icon_size: QSize = QSize(16, 16),
        parent: QWidget | None = None,
        icon_offset: QPoint | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("inlineIconButton")
        self.setFixedSize(24, 24)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        icon = AssetIcon(asset_name, icon_size, self, offset=icon_offset)
        icon.setFixedSize(icon_size)
        icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(icon, 1, Qt.AlignmentFlag.AlignCenter)


class ResizeHandle(QWidget):
    def __init__(self, edges: tuple[bool, bool, bool, bool], cursor: Qt.CursorShape, parent: QWidget) -> None:
        super().__init__(parent)
        self.edges = edges
        self._drag_start_pos: QPoint | None = None
        self._drag_start_geometry: QRect | None = None
        self.setCursor(cursor)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            return
        window = self.window()
        if not hasattr(window, "resize_from_handle") or not hasattr(window, "is_window_maximized"):
            return
        if window.is_window_maximized():
            return
        self._drag_start_pos = window._event_global_pos(event) if hasattr(window, "_event_global_pos") else event.globalPosition().toPoint()
        self._drag_start_geometry = window.geometry()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        window = self.window()
        if (
            not hasattr(window, "resize_from_handle")
            or self._drag_start_pos is None
            or self._drag_start_geometry is None
        ):
            return
        current_pos = window._event_global_pos(event) if hasattr(window, "_event_global_pos") else event.globalPosition().toPoint()
        window.resize_from_handle(self.edges, self._drag_start_geometry, self._drag_start_pos, current_pos)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag_start_pos = None
        self._drag_start_geometry = None
        event.accept()


class ChatInput(QTextEdit):
    submit_requested = Signal()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            event.accept()
            self.submit_requested.emit()
            return
        super().keyPressEvent(event)


class AiWorker(QObject):
    finished = Signal(object)

    def __init__(
        self,
        service: AiEqualizerService,
        text: str,
        preset: Preset,
        *,
        saved_presets: list[Preset],
        model_path: Path | None,
        device_curve: FrequencyCurve | None,
        chat_history: list[dict[str, str]],
    ) -> None:
        super().__init__()
        self.service = service
        self.text = text
        self.preset = preset
        self.saved_presets = saved_presets
        self.model_path = model_path
        self.device_curve = device_curve
        self.chat_history = chat_history

    def run(self) -> None:
        self.finished.emit(
            self.service.suggest_preset(
                self.text,
                self.preset,
                saved_presets=self.saved_presets,
                model_path=self.model_path,
                device_curve=self.device_curve,
                chat_history=self.chat_history,
            )
        )


class AutoEqWorker(QObject):
    finished = Signal(object, str, str, str)
    failed = Signal(str)

    def __init__(
        self,
        device_curve: FrequencyCurve,
        target_curve: FrequencyCurve,
        backend: str,
        mode_label: str,
    ) -> None:
        super().__init__()
        self.device_curve = device_curve
        self.target_curve = target_curve
        self.backend = backend
        self.mode_label = mode_label

    def run(self) -> None:
        try:
            result = build_autoeq_preset_result(
                self.device_curve,
                self.target_curve,
                backend=self.backend,
            )
        except Exception as exc:  # noqa: BLE001 - runs in a worker thread, report to UI.
            self.failed.emit(str(exc))
            return
        self.finished.emit(result, self.mode_label, self.device_curve.name, self.target_curve.name)


class FilterTypeIcon(QLabel):
    info_requested = Signal(str, QPoint)

    ICON_SIZE = QSize(20, 20)
    _pixmap_cache: dict[str, QPixmap] = {}
    _full_pixmap_cache: dict[str, QPixmap] = {}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._filter_type = "peaking"
        self.setFixedSize(24, 24)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_square_side(self, side: int) -> None:
        clean = max(20, int(side))
        self.setFixedSize(clean, clean)

    def set_filter_type(self, filter_type: str) -> None:
        self._filter_type = filter_type if filter_type in FILTER_TYPE_ICON_FILES else "peaking"
        self.update()

    def enterEvent(self, event) -> None:  # type: ignore[override]
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.info_requested.emit(self._filter_type, self.mapToGlobal(self.rect().topLeft()))
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hovered = self.underMouse()
        painter.setPen(QPen(QColor("#586171" if hovered else "#313640"), 1.0))
        painter.setBrush(QColor("#1f242c" if hovered else "#111318"))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        pixmap = self._pixmap_for_type(self._filter_type)
        if pixmap.isNull():
            return
        target = QRectF(
            (self.width() - self.ICON_SIZE.width()) / 2.0,
            (self.height() - self.ICON_SIZE.height()) / 2.0,
            self.ICON_SIZE.width(),
            self.ICON_SIZE.height(),
        )
        painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))

    @classmethod
    def _pixmap_for_type(cls, filter_type: str) -> QPixmap:
        if filter_type not in cls._pixmap_cache:
            cls._pixmap_cache[filter_type] = cls._scaled_icon_pixmap(cls.full_pixmap_for_type(filter_type))
        return cls._pixmap_cache[filter_type]

    @classmethod
    def full_pixmap_for_type(cls, filter_type: str) -> QPixmap:
        clean_type = filter_type if filter_type in FILTER_TYPE_ICON_FILES else "peaking"
        if clean_type not in cls._full_pixmap_cache:
            file_name = FILTER_TYPE_ICON_FILES.get(clean_type, FILTER_TYPE_ICON_FILES["peaking"])
            cls._full_pixmap_cache[clean_type] = QPixmap(str(resource_path(f"assets/{file_name}")))
        return cls._full_pixmap_cache[clean_type]

    @classmethod
    def _scaled_icon_pixmap(cls, pixmap: QPixmap) -> QPixmap:
        if pixmap.isNull():
            return pixmap
        source = cls._icon_source_rect(pixmap)
        cropped = pixmap.copy(source)
        dpr = current_device_pixel_ratio()
        pixel_size = QSize(
            max(1, int(round(cls.ICON_SIZE.width() * dpr))),
            max(1, int(round(cls.ICON_SIZE.height() * dpr))),
        )
        scaled = cropped.scaled(
            pixel_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        return scaled

    @staticmethod
    def _icon_source_rect(pixmap: QPixmap) -> QRect:
        width = pixmap.width()
        height = pixmap.height()
        side = int(min(width, height) * 0.92)
        left = int((width - side) / 2)
        top = int((height - side) / 2)
        return QRect(
            max(0, left),
            max(0, top),
            max(1, min(width - max(0, left), side)),
            max(1, min(height - max(0, top), side)),
        )


class ParamInfoLabel(QLabel):
    info_requested = Signal(str, object)

    def __init__(self, param_key: str, text: str) -> None:
        super().__init__(text)
        self.param_key = param_key
        self.setObjectName("paramLabel")
        self.setProperty("hovered", False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        super().enterEvent(event)
        self._set_hovered(True)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        self._set_hovered(False)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.info_requested.emit(self.param_key, self)
            event.accept()
            return
        super().mousePressEvent(event)

    def _set_hovered(self, hovered: bool) -> None:
        if self.property("hovered") == hovered:
            return
        self.setProperty("hovered", hovered)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class FilterIndexLabel(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: 0;")

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor("#303745"), 1.0))
        painter.setBrush(QColor("#202632"))
        painter.drawRoundedRect(rect, 5.0, 5.0)
        painter.setPen(QColor("#dce3ea"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())


class FilterEditorRow(QFrame):
    changed = Signal()
    selected = Signal(object)
    param_info_requested = Signal(str, object)
    FIXED_WIDTH = 458
    FIXED_HEIGHT = 122
    INDEX_WIDTH = 20
    TYPE_WIDTH = 70
    GAIN_DIAL_WIDTH = 58
    GAIN_SPIN_WIDTH = 78
    FREQ_WIDTH = 86
    Q_WIDTH = 76
    COLUMN_SPACING = 3

    def __init__(self, eq_filter: EqFilter, index: int) -> None:
        super().__init__()
        self.index = index
        self._syncing = False
        self.setObjectName("filterRow")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setFixedSize(self.FIXED_WIDTH, self.FIXED_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(self.COLUMN_SPACING)
        layout.setVerticalSpacing(5)

        self.enabled_check = AieqCheckBox()
        self.enabled_check.setToolTip("Вкл")
        self.enabled_check.setFixedSize(24, 24)
        self.enabled_check.setChecked(eq_filter.enabled)
        self.type_combo = SimpleComboBox()
        self.type_combo.setProperty("enableCollapsedWheel", True)
        self.type_combo.setProperty("popupVisibleRows", 3)
        self.type_combo.setProperty("popupRowHeight", POPUP_OPTION_DEFAULT_ROW_HEIGHT)
        self.type_combo.setProperty("alignPopupCenter", True)
        self.set_filter_type_labels({}, current_type=eq_filter.type)
        self.type_combo.setFixedWidth(self.TYPE_WIDTH)
        self.type_icon = FilterTypeIcon(self)
        self.type_icon.set_filter_type(eq_filter.type)

        self.gain_dial = QDial()
        self.gain_dial.setRange(-2400, 2400)
        self.gain_dial.setSingleStep(25)
        self.gain_dial.setPageStep(100)
        self.gain_dial.setNotchesVisible(True)
        self.gain_dial.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        dial_palette = self.gain_dial.palette()
        dial_palette.setColor(QPalette.ColorRole.Highlight, QColor("#303745"))
        accent_role = getattr(QPalette.ColorRole, "Accent", None)
        if accent_role is not None:
            dial_palette.setColor(accent_role, QColor("#303745"))
        self.gain_dial.setPalette(dial_palette)
        self.gain_dial.setFixedSize(self.GAIN_DIAL_WIDTH, 58)
        self.gain_dial.setValue(int(round(eq_filter.gain * 100)))

        self.gain_spin = CleanDoubleSpinBox()
        self.gain_spin.setRange(-24.0, 24.0)
        self.gain_spin.setDecimals(2)
        self.gain_spin.setSingleStep(0.25)
        self.gain_spin.setKeyboardTracking(False)
        self.gain_spin.setValue(eq_filter.gain)
        self.gain_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gain_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.gain_spin.setFixedWidth(self.GAIN_SPIN_WIDTH)

        self.freq_spin = CleanDoubleSpinBox()
        self.freq_spin.setRange(20.0, 20000.0)
        self.freq_spin.setDecimals(0)
        self.freq_spin.setSingleStep(10.0)
        self.freq_spin.setKeyboardTracking(False)
        self.freq_spin.setValue(eq_filter.freq)
        self.freq_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.freq_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.freq_spin.setFixedWidth(self.FREQ_WIDTH)

        self.q_spin = CleanDoubleSpinBox()
        self.q_spin.setRange(0.1, 18.0)
        self.q_spin.setDecimals(3)
        self.q_spin.setSingleStep(0.01)
        self.q_spin.setKeyboardTracking(False)
        self.q_spin.setValue(eq_filter.q)
        self.q_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.q_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.q_spin.setFixedWidth(self.Q_WIDTH)

        self.value_editors = (self.gain_spin, self.freq_spin, self.q_spin)
        for focus_reset_widget in (self.enabled_check, self.type_combo, self.gain_dial):
            focus_reset_widget.installEventFilter(self)

        self.index_label = FilterIndexLabel(f"{index + 1:02d}")
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.index_label.setFixedWidth(self.INDEX_WIDTH)
        layout.setColumnMinimumWidth(0, self.INDEX_WIDTH)
        layout.setColumnMinimumWidth(1, 24)
        layout.setColumnMinimumWidth(2, self.TYPE_WIDTH)
        layout.setColumnMinimumWidth(3, self.GAIN_DIAL_WIDTH)
        layout.setColumnMinimumWidth(4, self.GAIN_SPIN_WIDTH)
        layout.setColumnMinimumWidth(5, self.FREQ_WIDTH)
        layout.setColumnMinimumWidth(6, self.Q_WIDTH)
        layout.setRowMinimumHeight(0, 24)
        layout.setRowMinimumHeight(1, 62)
        layout.addWidget(self.index_label, 0, 0, 2, 1)
        layout.addWidget(self.enabled_check, 0, 1, Qt.AlignmentFlag.AlignCenter)
        self.type_label = self._param_label("type", self.TYPE_WIDTH)
        self.gain_label = self._param_label("gain", self.GAIN_DIAL_WIDTH + self.COLUMN_SPACING + self.GAIN_SPIN_WIDTH)
        self.freq_label = self._param_label("freq", self.FREQ_WIDTH)
        self.q_label = self._param_label("q", self.Q_WIDTH)

        layout.addWidget(self.type_label, 0, 2)
        layout.addWidget(self.type_combo, 1, 2)
        layout.addWidget(self.gain_label, 0, 3, 1, 2)
        layout.addWidget(self.gain_dial, 1, 3)
        layout.addWidget(self.gain_spin, 1, 4)
        layout.addWidget(self.freq_label, 0, 5)
        layout.addWidget(self.freq_spin, 1, 5)
        layout.addWidget(self.q_label, 0, 6)
        layout.addWidget(self.q_spin, 1, 6)

        self.enabled_check.toggled.connect(self._emit_changed)
        self.type_combo.currentIndexChanged.connect(self.sync_type_icon)
        self.type_combo.currentIndexChanged.connect(self._emit_changed)
        self.freq_spin.valueChanged.connect(self._emit_changed)
        self.q_spin.valueChanged.connect(self._emit_changed)
        self.gain_spin.valueChanged.connect(self._on_gain_spin_changed)
        self.gain_spin.editingFinished.connect(self._sync_gain_from_spin)
        self.gain_dial.valueChanged.connect(self._on_gain_dial_changed)
        QTimer.singleShot(0, self.position_type_icon)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.position_type_icon()

    def position_type_icon(self) -> None:
        if not hasattr(self, "type_icon") or not hasattr(self, "type_combo") or not hasattr(self, "index_label"):
            return
        type_rect = self.type_combo.geometry()
        index_rect = self.index_label.geometry()
        side = type_rect.height() if type_rect.height() > 0 else self.type_combo.sizeHint().height()
        self.type_icon.set_square_side(side)
        left_edge = index_rect.x() + index_rect.width()
        if hasattr(self, "gain_spin") and hasattr(self, "freq_spin"):
            gain_rect = self.gain_spin.geometry()
            freq_rect = self.freq_spin.geometry()
            target_gap = max(0, freq_rect.x() - (gain_rect.x() + gain_rect.width()))
        else:
            target_gap = self.COLUMN_SPACING
        x = max(left_edge, type_rect.x() - side - target_gap)
        y = type_rect.y()
        self.type_icon.move(x, y)
        self.type_icon.raise_()

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if event.type() in {
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonDblClick,
            QEvent.Type.Wheel,
            QEvent.Type.FocusIn,
        }:
            self.clear_value_editor_focus()
        return super().eventFilter(obj, event)

    def clear_value_editor_focus(self) -> None:
        for editor in self.value_editors:
            editor.clear_edit_focus()

    def _param_label(self, param_key: str, width: int) -> QLabel:
        label = ParamInfoLabel(param_key, param_key)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedSize(width, 24)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        label.info_requested.connect(self.param_info_requested)
        return label

    def apply_labels(
        self,
        labels: dict[str, str],
        enabled_tooltip: str,
        filter_type_labels: dict[str, str] | None = None,
    ) -> None:
        self.set_filter_type_labels(filter_type_labels or {})
        self._set_param_label_text(self.type_label, labels.get("type", "type"))
        self._set_param_label_text(self.gain_label, labels.get("gain", "gain"))
        self._set_param_label_text(self.freq_label, labels.get("freq", "freq"))
        self._set_param_label_text(self.q_label, labels.get("q", "q"))
        self.enabled_check.setToolTip(enabled_tooltip)

    def set_filter_type_labels(self, labels: dict[str, str], *, current_type: str | None = None) -> None:
        selected_type = current_type or self.selected_filter_type()
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        selected_index = 0
        for index, filter_type in enumerate(FILTER_TYPES):
            self.type_combo.addItem(labels.get(filter_type, filter_type), filter_type)
            if filter_type == selected_type:
                selected_index = index
        self.type_combo.setCurrentIndex(selected_index)
        self.type_combo.blockSignals(False)
        if hasattr(self, "type_icon"):
            self.sync_type_icon()

    def selected_filter_type(self) -> str:
        value = self.type_combo.currentData()
        if value is None:
            return self.type_combo.currentText()
        return str(value)

    def sync_type_icon(self, *_args) -> None:
        self.type_icon.set_filter_type(self.selected_filter_type())

    def _set_param_label_text(self, label: QLabel, text: str) -> None:
        label.setText(text)
        label.setToolTip(text)
        font = label.font()
        font.setPixelSize(FILTER_LABEL_FONT_PX)
        label.setFont(font)
        label.style().unpolish(label)
        label.style().polish(label)
        label.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.selected.emit(self)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def to_filter(self) -> EqFilter:
        return EqFilter(
            self.selected_filter_type(),
            self.freq_spin.value(),
            self.q_spin.value(),
            self.gain_spin.value(),
            self.enabled_check.isChecked(),
        ).sanitized()

    def _on_gain_spin_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.gain_dial.setValue(int(round(value * 100)))
        self.gain_dial.update()
        self._syncing = False
        self._emit_changed()

    def _sync_gain_from_spin(self) -> None:
        self._on_gain_spin_changed(self.gain_spin.value())

    def _on_gain_dial_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.gain_spin.setValue(value / 100.0)
        self._syncing = False
        self._emit_changed()

    def _emit_changed(self) -> None:
        self.selected.emit(self)
        self.changed.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AIEQ")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.store = PresetStore()
        self.chat_store = ChatStore()
        self.ai_service = AiEqualizerService()
        self.audio_engine = AudioEngine()
        self.current_preset = flat_preset()
        self._clean_preset_filter_signature = self.preset_filter_signature(self.current_preset)
        self._clean_preset_id: int | None = self.current_preset.id
        self._clean_preset_name = self.current_preset.name
        self.saved_presets: list[Preset] = []
        self.chat_sessions: list[ChatSession] = []
        self.current_chat_id: int | None = None
        self.current_chat_context_full = False
        self.compare_ids: set[int] = set()
        self.device_curves: list[FrequencyCurve] = []
        self.target_curves: list[FrequencyCurve] = []
        self.target_options: list[FrequencyCurve] = []
        self.selected_device_curve: FrequencyCurve | None = None
        self.input_devices: list[AudioDevice] = []
        self.output_devices: list[AudioDevice] = []
        self.audio_settings: list[AudioStreamSetting] = []
        self._updating = False
        self._ai_thread: QThread | None = None
        self._ai_worker: AiWorker | None = None
        self._autoeq_thread: QThread | None = None
        self._autoeq_worker: AutoEqWorker | None = None
        self._chat_popup: QWidget | None = None
        self._filter_type_popup: QWidget | None = None
        self._filter_param_popup: QWidget | None = None
        self._tooltip_reopen_blocked = False
        self._title_drag_pos: QPointF | None = None
        self._compare_popup: QWidget | None = None
        self._resize_margin = 8
        self._window_maximized = False
        self._normal_geometry: QRect | None = None
        self._startup_layout_finalized = False
        self.resize_handles: list[ResizeHandle] = []
        self.chat_messages: list[dict[str, str]] = []
        self.filter_rows: list[FilterEditorRow] = []
        self.selected_filter_row = -1
        self.model_context_limits: dict[str, int | None] = {}
        self._ai_runtime_signature: tuple[int, int, float] | None = None
        self.settings = QSettings()
        self.language_options = list_language_options()
        self.language_code = str(self.settings.value("app/language", DEFAULT_LANGUAGE_CODE) or DEFAULT_LANGUAGE_CODE)
        self.language_name = ""
        self.translations: dict[str, str] = {}
        self.load_language(self.language_code)
        self.audio_update_timer = QTimer(self)
        self.audio_update_timer.setSingleShot(True)
        self.audio_update_timer.timeout.connect(self.apply_audio_preset)
        self.toast_timer = QTimer(self)
        self.toast_timer.setSingleShot(True)
        self.toast_timer.timeout.connect(self.hide_toast)
        self.audio_latency_timer = QTimer(self)
        self.audio_latency_timer.timeout.connect(self.update_audio_latency_label)

        ensure_curve_dirs()
        self._build_ui()
        self._apply_style()
        self.refresh_languages(show_feedback=False)
        self.refresh_devices()
        self.refresh_ai_models()
        self.refresh_chat_sessions()
        self.start_new_chat(show_feedback=False)
        self.refresh_curve_lists()
        self.refresh_presets()
        self.populate_filter_editor()
        self.update_graph()
        self.restore_window_layout()
        self.schedule_filter_container_sync()
        QTimer.singleShot(0, self.position_title_logo)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast(self.t("toast.wait_ai"))
            event.ignore()
            return
        if self._autoeq_thread is not None and self._autoeq_thread.isRunning():
            self.show_toast(self.t("toast.autoeq_still_running"))
            event.ignore()
            return
        self.save_window_layout()
        self.audio_engine.stop()
        self.audio_latency_timer.stop()
        self.ai_service.shutdown()
        super().closeEvent(event)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._startup_layout_finalized:
            return
        self._startup_layout_finalized = True
        QTimer.singleShot(0, self.finalize_startup_layout)
        QTimer.singleShot(80, self.finalize_startup_layout)

    def finalize_startup_layout(self) -> None:
        if self.is_window_maximized() or self.isFullScreen() or self.isMinimized():
            return
        if self._is_screen_wide_geometry(self.geometry()):
            default_geometry = self.default_window_geometry()
            if not self._is_screen_wide_geometry(default_geometry):
                self.setGeometry(default_geometry)
        self.clamp_window_to_available_screen()
        self.center_window_on_available_screen()
        if not self._is_screen_wide_geometry(self.geometry()):
            self._normal_geometry = self.geometry()

    def load_language(self, code: str) -> None:
        language_code, language_name, translations = load_language_bundle(code)
        self.language_code = language_code
        self.language_name = language_name
        self.translations = translations

    def t(self, key: str) -> str:
        return self.translations.get(key, key)

    def set_language(self, code: str) -> None:
        if not code or code == self.language_code:
            return
        self.load_language(code)
        self.settings.setValue("app/language", self.language_code)
        self.apply_language()
        self.render_chat_messages()

    def refresh_language_combo(self) -> None:
        if not hasattr(self, "language_combo"):
            return
        self.language_options = list_language_options()
        self.language_combo.blockSignals(True)
        self.language_combo.clear()
        selected_index = 0
        for index, (code, name) in enumerate(self.language_options):
            self.language_combo.addItem(name, code)
            if code == self.language_code:
                selected_index = index
        self.language_combo.setCurrentIndex(selected_index)
        self.language_combo.blockSignals(False)

    def refresh_languages(self, *, show_feedback: bool = False) -> None:
        self.language_options = list_language_options()
        self.load_language(self.language_code)
        self.refresh_language_combo()
        self.apply_language()
        if show_feedback:
            self.show_toast(self.t("toast.language_lists_updated"))

    def on_language_changed(self, _index: int) -> None:
        code = self.language_combo.currentData() if hasattr(self, "language_combo") else None
        if code:
            self.set_language(str(code))

    def resize_settings_panel(self) -> None:
        if not hasattr(self, "settings_panel"):
            return
        self.settings_panel.adjustSize()
        self.settings_panel.setFixedWidth(SETTINGS_PANEL_WIDTH)
        self.settings_panel.update_rounded_mask()

    def set_wrapped_checkbox_text(self, checkbox: AieqCheckBox, text: str, *, width: int = 360) -> None:
        metrics = QFontMetricsF(checkbox.font())
        words = text.split()
        if not words:
            checkbox.setText(text)
            checkbox.setFixedHeight(24)
            return
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if metrics.horizontalAdvance(candidate) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        checkbox.setText("\n".join(lines))
        checkbox.setFixedHeight(max(24, int(metrics.lineSpacing() * len(lines) + 8)))

    def apply_language(self) -> None:
        search_text = self.t("common.search")
        for combo, empty_key in (
            (getattr(self, "input_combo", None), "empty.inputs"),
            (getattr(self, "output_combo", None), "empty.outputs"),
            (getattr(self, "sample_rate_combo", None), "empty.sample_rates"),
            (getattr(self, "audio_dtype_combo", None), "empty.formats"),
            (getattr(self, "device_curve_combo", None), "empty.devices"),
            (getattr(self, "current_selector", None), "empty.presets"),
            (getattr(self, "ai_model_combo", None), "empty.models"),
            (getattr(self, "target_curve_combo", None), "empty.curves"),
            (getattr(self, "language_combo", None), "settings.no_languages"),
        ):
            if isinstance(combo, SearchableComboBox):
                combo.empty_text = self.t(empty_key)
                combo.search_text = search_text

        if hasattr(self, "settings_button"):
            self.settings_button.setToolTip(self.t("settings.title"))
        if hasattr(self, "github_button"):
            self.github_button.setToolTip(self.t("tooltip.github"))
        if hasattr(self, "input_label"):
            self.input_label.setText(self.t("top.input"))
            self.output_label.setText(self.t("top.output"))
            self.sample_rate_label.setText(self.t("top.sample_rate"))
            self.audio_dtype_label.setText(self.t("top.format"))
            self.refresh_devices_button.setToolTip(self.t("tooltip.refresh_devices"))
            self.audio_button.setToolTip(self.t("audio.stop") if self.audio_engine.is_running else self.t("audio.start"))

        if hasattr(self, "settings_audio_title"):
            self.settings_audio_title.setText(self.t("settings.audio"))
            self.current_latency_label.setText(self.t("settings.current_latency"))
            self.set_wrapped_checkbox_text(self.custom_latency_check, self.t("settings.custom_latency"))
            self.custom_latency_label.setText(self.t("settings.latency_ms"))
            self.settings_ai_title.setText(self.t("settings.ai"))
            self.set_wrapped_checkbox_text(self.allow_cpu_fallback_check, self.t("settings.allow_cpu"))
            self.set_wrapped_checkbox_text(self.advanced_ai_check, self.t("settings.advanced_ai"))
            self.ai_ctx_label.setText(self.t("settings.context"))
            self.ai_max_tokens_label.setText(self.t("settings.tokens"))
            self.ai_temperature_label.setText(self.t("settings.temperature"))
            self.settings_app_title.setText(self.t("settings.app"))
            self.language_label.setText(self.t("settings.language"))
            self.refresh_languages_button.setToolTip(self.t("tooltip.refresh_languages"))
            self.refresh_language_combo()
            self.refresh_ai_settings_limits()

        if hasattr(self, "graph_group"):
            self.graph_group.setTitle(self.t("section.graph"))
            self.plot.setLabel("bottom", self.t("graph.frequency"))
            self.plot.setLabel("left", self.t("graph.gain"))
            self.plot_legend.setItemName(self.device_curve_item, self.t("graph.device"))
            self.plot_legend.setItemName(self.current_curve, self.t("graph.current"))
            self.plot_legend.setItemName(self.target_curve_item, self.t("graph.target"))
            self.device_curve_label.setText(self.t("label.device"))
            self.current_preset_label.setText(self.t("label.current_preset"))
            self.compare_button.setText(self.t("button.compare"))
            self.import_button.setText(self.t("button.import"))
            self.export_button.setText(self.t("button.export"))
            self.refresh_curves_button.setToolTip(self.t("tooltip.refresh_lists"))

        if hasattr(self, "filters_group"):
            self.filters_group.setTitle(self.t("section.filters"))
            self.add_filter_button.setText(self.t("button.add"))
            self.remove_filter_button.setText(self.t("button.remove"))
            self.clear_filters_button.setText(self.t("button.clear"))
            self.delete_preset_button.setText(self.t("button.delete_preset"))
            self.save_button.setText(self.t("button.save_preset"))

        if hasattr(self, "assistant_group"):
            self.assistant_group.setTitle(self.t("section.assistant"))
            self.ai_model_label.setText(self.t("label.model"))
            self.refresh_models_button.setToolTip(self.t("tooltip.refresh_models"))
            self.chat_menu_button.setToolTip(self.t("tooltip.saved_chats"))
            self.delete_chat_button.setToolTip(self.t("tooltip.delete_chat"))
            self.new_chat_button.setToolTip(self.t("tooltip.new_chat"))
            self.send_button.setToolTip(self.t("tooltip.send"))
            self.target_curve_label.setText(self.t("label.target_curve"))
            self.show_target_checkbox.setText(self.t("checkbox.show_target"))
            self.autoeq_algorithm_label.setText(self.t("label.algorithm"))
            self.run_autoeq_button.setText(self.t("button.run_autoeq"))
            self.side_tabs.setTabText(0, self.t("tab.ai_chat"))
            self.side_tabs.setTabText(1, self.t("tab.autoeq"))
            self.update_chat_context_state()

        self.apply_filter_row_labels()
        self.refresh_localized_combo_items()
        if hasattr(self, "plot_legend"):
            self.update_graph()
        if hasattr(self, "settings_panel"):
            self.resize_settings_panel()

    def apply_filter_row_labels(self) -> None:
        labels = {
            "type": self.t("filter.type"),
            "gain": self.t("filter.gain"),
            "freq": self.t("filter.freq"),
            "q": self.t("filter.q"),
        }
        filter_type_labels = {filter_type: self.filter_type_label(filter_type) for filter_type in FILTER_TYPES}
        for row in getattr(self, "filter_rows", []):
            row.apply_labels(labels, self.t("filter.enabled"), filter_type_labels)

    def filter_type_label(self, filter_type: str) -> str:
        key = f"filter_type.{filter_type}"
        translated = self.t(key)
        return filter_type if translated == key else translated

    def filter_type_description(self, filter_type: str) -> str:
        key = f"filter_type_desc.{filter_type}"
        translated = self.t(key)
        return "" if translated == key else translated

    def filter_param_description(self, param_key: str) -> str:
        key = f"filter_param_desc.{param_key}"
        translated = self.t(key)
        return "" if translated == key else translated

    def device_curve_display_name(self, curve: FrequencyCurve) -> str:
        if curve.name == "Default" and curve.path is None:
            return self.t("device.default")
        return curve.name

    def new_preset_display_name(self) -> str:
        return self.t("preset.new")

    def refresh_localized_combo_items(self) -> None:
        if hasattr(self, "device_curve_combo"):
            for row in range(self.device_curve_combo.count()):
                curve_index = self.device_curve_combo.itemData(row)
                if isinstance(curve_index, int) and 0 <= curve_index < len(self.device_curves):
                    self.device_curve_combo.setItemText(row, self.device_curve_display_name(self.device_curves[curve_index]))
        if hasattr(self, "current_selector"):
            for row in range(self.current_selector.count()):
                if self.current_selector.itemData(row) == NEW_PRESET_ID:
                    self.current_selector.setItemText(row, self.new_preset_display_name())

    def _filter_tooltip_open(self) -> bool:
        return any(
            popup is not None and popup.isVisible()
            for popup in (self._filter_type_popup, self._filter_param_popup)
        )

    def _can_open_filter_tooltip(self) -> bool:
        return not self._tooltip_reopen_blocked and not self._filter_tooltip_open()

    def _on_filter_tooltip_destroyed(self, attr_name: str) -> None:
        setattr(self, attr_name, None)
        self._tooltip_reopen_blocked = True
        QTimer.singleShot(180, self._release_filter_tooltip_block)

    def _release_filter_tooltip_block(self) -> None:
        self._tooltip_reopen_blocked = False

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("windowRoot")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.window_frame = QFrame()
        self.window_frame.setObjectName("windowFrame")
        frame_layout = QVBoxLayout(self.window_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        outer.addWidget(self.window_frame)

        frame_layout.addWidget(self._build_window_title_bar())

        content = QWidget()
        content.setObjectName("contentRoot")
        root = QVBoxLayout(content)
        root.setContentsMargins(14, 10, 14, 14)
        root.setSpacing(10)

        root.addWidget(self._build_top_bar())

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self._build_left_panel())
        self.main_splitter.addWidget(self._build_chat_panel())
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        root.addWidget(self.main_splitter, 1)
        frame_layout.addWidget(content, 1)

        self.build_resize_handles(central)
        self.setCentralWidget(central)

        self.toast_label = QLabel(central)
        self.toast_label.setObjectName("toast")
        self.toast_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.toast_label.hide()

    def build_resize_handles(self, parent: QWidget) -> None:
        specs = [
            ((True, False, False, False), Qt.CursorShape.SizeHorCursor),
            ((False, True, False, False), Qt.CursorShape.SizeHorCursor),
            ((False, False, True, False), Qt.CursorShape.SizeVerCursor),
            ((False, False, False, True), Qt.CursorShape.SizeVerCursor),
        ]
        self.resize_handles = [ResizeHandle(edges, cursor, parent) for edges, cursor in specs]

    def _build_window_title_bar(self) -> QWidget:
        title_bar = QWidget()
        self.window_title_bar = title_bar
        self.title_drag_widgets: set[QObject] = {title_bar}
        title_bar.setObjectName("windowTitleBar")
        title_bar.setFixedHeight(TITLE_BAR_HEIGHT)
        title_bar.installEventFilter(self)

        layout = QGridLayout(title_bar)
        layout.setContentsMargins(10, 3, 8, 3)
        layout.setHorizontalSpacing(6)

        self.title_logo = QLabel()
        self.title_logo.setObjectName("windowTitleLogo")
        self.title_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_logo.installEventFilter(self)
        self.title_drag_widgets.add(self.title_logo)
        logo_pixmap = load_asset_pixmap("logo.png", QSize(98, 19))
        if not logo_pixmap.isNull():
            self.title_logo.setPixmap(logo_pixmap)
        self.title_logo.setFixedHeight(30)
        self.title_logo.setContentsMargins(0, 0, 0, 0)

        left_controls = QWidget()
        left_controls.setObjectName("windowControls")
        left_controls.setFixedWidth(TITLE_CONTROLS_WIDTH)
        left_controls_layout = QHBoxLayout(left_controls)
        left_controls_layout.setContentsMargins(0, 0, 0, 0)
        left_controls_layout.setSpacing(0)
        self.settings_button = QPushButton("⚙︎")
        self.settings_button.setObjectName("settingsIconButton")
        self.settings_button.setToolTip(self.t("settings.title"))
        self.settings_button.setFixedSize(22, 24)
        self.settings_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.settings_button.clicked.connect(self.show_settings_menu)
        self.settings_menu = QMenu(self.settings_button)
        self.settings_menu.setObjectName("settingsMenu")
        self._build_settings_menu()
        left_controls_layout.addWidget(self.settings_button, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        left_controls_layout.addSpacing(6)
        self.github_button = InlineAssetButton("github.svg", QSize(20, 20))
        self.github_button.setObjectName("settingsIconButton")
        self.github_button.setToolTip(self.t("tooltip.github"))
        self.github_button.setFixedSize(22, 24)
        self.github_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.github_button.clicked.connect(self.open_github_repository)
        left_controls_layout.addWidget(self.github_button, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        left_controls_layout.addStretch(1)

        controls = QWidget()
        controls.setObjectName("windowControls")
        controls.setFixedWidth(TITLE_CONTROLS_WIDTH)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(3)
        self.window_minimize_button = WindowControlButton("\u2212")
        self.window_maximize_button = WindowControlButton("\u25a1")
        self.window_close_button = WindowControlButton("\u00d7", close_button=True)
        for button in (self.window_minimize_button, self.window_maximize_button, self.window_close_button):
            button.setObjectName("windowControlButton")
            controls_layout.addWidget(button)
        self.window_close_button.setObjectName("windowCloseButton")
        self.window_minimize_button.clicked.connect(self.showMinimized)
        self.window_maximize_button.clicked.connect(self.toggle_window_maximized)
        self.window_close_button.clicked.connect(self.close)

        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.addWidget(left_controls, 0, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.title_logo, 0, 1, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(controls, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return title_bar

    def position_title_logo(self) -> None:
        if not hasattr(self, "title_logo") or not hasattr(self, "output_combo"):
            return
        output_top = self.output_combo.mapTo(self, QPoint(0, 0)).y()
        if output_top <= 0:
            return
        label_top = self.title_logo.mapTo(self, QPoint(0, 0)).y()
        pixmap = self.title_logo.pixmap()
        pixmap_height = pixmap.deviceIndependentSize().height() if pixmap is not None and not pixmap.isNull() else 19.0
        desired_center = output_top / 2.0
        top_margin = int(round(desired_center - label_top - pixmap_height / 2.0))
        top_margin = max(0, min(max(0, self.title_logo.height() - pixmap_height), top_margin))
        current = self.title_logo.contentsMargins()
        if current.top() != top_margin:
            self.title_logo.setContentsMargins(0, top_margin, 0, 0)
            self.title_logo.update()

    def toggle_window_maximized(self) -> None:
        if self.is_window_maximized():
            self.restore_window_from_maximized()
            return
        self.maximize_window_to_available_screen()

    def is_window_maximized(self) -> bool:
        return self._window_maximized

    def maximize_window_to_available_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        self._normal_geometry = self.geometry()
        self._window_maximized = True
        self.setGeometry(screen.availableGeometry())
        self.update_window_chrome()

    def restore_window_from_maximized(self) -> None:
        geometry = self._normal_geometry
        self._window_maximized = False
        if geometry is not None and not geometry.isNull():
            self.setGeometry(geometry)
        else:
            self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
            self.clamp_window_to_available_screen()
        self.center_window_on_available_screen()
        self.update_window_chrome()

    def update_window_chrome(self) -> None:
        maximized = self.is_window_maximized()
        if hasattr(self, "window_maximize_button"):
            self.window_maximize_button.set_symbol("\u2750" if maximized else "\u25a1")
        if hasattr(self, "window_frame"):
            self.window_frame.setProperty("maximized", maximized)
            self.window_frame.style().unpolish(self.window_frame)
            self.window_frame.style().polish(self.window_frame)
        self.position_resize_handles()
        self.update_window_mask()

    def update_window_mask(self) -> None:
        if self.is_window_maximized() or self.isFullScreen():
            self.clearMask()
            return
        rect = QRectF(self.rect())
        if rect.isEmpty():
            return
        path = QPainterPath()
        path.addRoundedRect(rect, WINDOW_RADIUS, WINDOW_RADIUS)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and not self.isMinimized():
            self.update_window_chrome()

    @staticmethod
    def _event_global_pos(event) -> QPoint:
        return event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()

    def handle_title_bar_event(self, event) -> bool:
        if event.type() == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
            self.toggle_window_maximized()
            return True
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._title_drag_pos = self._event_global_pos(event) - self.frameGeometry().topLeft()
            return True
        if event.type() == QEvent.Type.MouseMove and self._title_drag_pos is not None:
            if event.buttons() & Qt.MouseButton.LeftButton:
                global_pos = self._event_global_pos(event)
                if self.is_window_maximized():
                    width_before = max(1, self.width())
                    ratio = min(max((global_pos.x() - self.frameGeometry().x()) / width_before, 0.0), 1.0)
                    self.restore_window_from_maximized()
                    self._title_drag_pos = QPoint(int(self.width() * ratio), TITLE_BAR_HEIGHT // 2)
                self.move(global_pos - self._title_drag_pos)
                return True
        if event.type() == QEvent.Type.MouseButtonRelease:
            if not self.is_window_maximized():
                self._normal_geometry = self.geometry()
            self._title_drag_pos = None
            return True
        return False

    def nativeEvent(self, event_type, message):  # type: ignore[override]
        if sys.platform != "win32" or self.is_window_maximized() or self.isFullScreen():
            return super().nativeEvent(event_type, message)
        if event_type not in {"windows_generic_MSG", "windows_dispatcher_MSG"}:
            return super().nativeEvent(event_type, message)
        try:
            import ctypes
            import ctypes.wintypes

            msg = ctypes.wintypes.MSG.from_address(int(message))
        except Exception:
            return super().nativeEvent(event_type, message)

        wm_nchittest = 0x0084
        if msg.message != wm_nchittest:
            return super().nativeEvent(event_type, message)

        x = ctypes.c_short(msg.lParam & 0xFFFF).value
        y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
        frame = self.frameGeometry()
        margin = self._resize_margin
        left = frame.left() <= x < frame.left() + margin
        right = frame.right() - margin < x <= frame.right()
        top = frame.top() <= y < frame.top() + margin
        bottom = frame.bottom() - margin < y <= frame.bottom()

        htleft = 10
        htright = 11
        httop = 12
        htbottom = 15
        if left:
            return True, htleft
        if right:
            return True, htright
        if top:
            return True, httop
        if bottom:
            return True, htbottom
        return super().nativeEvent(event_type, message)

    def position_resize_handles(self) -> None:
        if not hasattr(self, "resize_handles") or not self.resize_handles:
            return
        margin = self._resize_margin
        width = self.width()
        height = self.height()
        geometries = [
            QRect(0, margin, margin, max(0, height - margin * 2)),
            QRect(max(0, width - margin), margin, margin, max(0, height - margin * 2)),
            QRect(margin, 0, max(0, width - margin * 2), margin),
            QRect(margin, max(0, height - margin), max(0, width - margin * 2), margin),
        ]
        visible = not self.is_window_maximized()
        for handle, geometry in zip(self.resize_handles, geometries, strict=False):
            handle.setGeometry(geometry)
            handle.setVisible(visible)
            handle.raise_()

    def resize_from_handle(
        self,
        edges: tuple[bool, bool, bool, bool],
        start_geometry: QRect,
        start_pos: QPoint,
        current_pos: QPoint,
    ) -> None:
        left_edge, right_edge, top_edge, bottom_edge = edges
        delta = current_pos - start_pos
        geometry = QRect(start_geometry)
        min_width = max(self.minimumWidth(), 640)
        min_height = max(self.minimumHeight(), 420)
        if left_edge:
            geometry.setLeft(min(geometry.left() + delta.x(), geometry.right() - min_width + 1))
        if right_edge:
            geometry.setRight(max(geometry.right() + delta.x(), geometry.left() + min_width - 1))
        if top_edge:
            geometry.setTop(min(geometry.top() + delta.y(), geometry.bottom() - min_height + 1))
        if bottom_edge:
            geometry.setBottom(max(geometry.bottom() + delta.y(), geometry.top() + min_height - 1))
        self.setGeometry(geometry)
        self._normal_geometry = self.geometry()

    def restore_window_layout(self) -> None:
        reset_layout = os.environ.get("AIEQ_RESET_WINDOW_LAYOUT", "").strip().casefold() in {"1", "true", "yes", "on"}
        if reset_layout:
            self.reset_window_layout_settings()
        normal_geometry = self._settings_rect("window/normal_geometry")
        if normal_geometry is not None and self._is_screen_wide_geometry(normal_geometry):
            normal_geometry = None
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        splitter_state = self.settings.value("window/main_splitter")
        if normal_geometry is not None and not reset_layout:
            self.setGeometry(normal_geometry)
            if state:
                self.restoreState(state)
        elif geometry and not reset_layout:
            self.restoreGeometry(geometry)
            if self._is_screen_wide_geometry(self.geometry()):
                self.apply_default_window_geometry()
            elif state:
                self.restoreState(state)
        else:
            self.apply_default_window_geometry()
        if splitter_state and not reset_layout:
            self.main_splitter.restoreState(splitter_state)
        else:
            self.apply_default_splitter_sizes()
        self.clamp_window_to_available_screen()
        self.center_window_on_available_screen()
        self._normal_geometry = self.geometry()
        self.update_window_chrome()

    def reset_window_layout_settings(self) -> None:
        for key in ("window/geometry", "window/normal_geometry", "window/state", "window/main_splitter"):
            self.settings.remove(key)
        self.settings.sync()

    def _settings_rect(self, key: str) -> QRect | None:
        value = self.settings.value(key)
        if isinstance(value, QRect) and not value.isNull():
            return QRect(value)
        return None

    def _is_screen_wide_geometry(self, geometry: QRect) -> bool:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None or geometry.isNull():
            return False
        available = screen.availableGeometry()
        tolerance = 2
        return geometry.width() >= available.width() - tolerance

    def default_window_geometry(self) -> QRect:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return QRect(0, 0, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        available = screen.availableGeometry()
        width = min(DEFAULT_WINDOW_WIDTH, available.width())
        height = min(DEFAULT_WINDOW_HEIGHT, available.height())
        geometry = QRect(0, 0, width, height)
        geometry.moveCenter(available.center())
        return geometry

    def apply_default_window_geometry(self) -> None:
        self.setGeometry(self.default_window_geometry())

    def apply_default_splitter_sizes(self) -> None:
        sizes = list(DEFAULT_SPLITTER_SIZES)
        self.main_splitter.setSizes(sizes)
        QTimer.singleShot(0, lambda sizes=sizes: self.main_splitter.setSizes(list(sizes)))

    def save_window_layout(self) -> None:
        if self.is_window_maximized():
            normal_geometry = QRect(self._normal_geometry) if self._normal_geometry is not None else QRect()
            if normal_geometry.isNull():
                normal_geometry = QRect(0, 0, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
            self.settings.setValue("window/normal_geometry", normal_geometry)
            self.settings.remove("window/geometry")
        else:
            current_geometry = QRect(self.geometry())
            previous_geometry = QRect(self._normal_geometry) if self._normal_geometry is not None else QRect()
            if self._is_screen_wide_geometry(current_geometry):
                if not previous_geometry.isNull() and not self._is_screen_wide_geometry(previous_geometry):
                    normal_geometry = previous_geometry
                else:
                    normal_geometry = self.default_window_geometry()
                self.settings.remove("window/geometry")
            else:
                normal_geometry = current_geometry
                self._normal_geometry = QRect(normal_geometry)
                self.settings.setValue("window/geometry", self.saveGeometry())
            self.settings.setValue("window/normal_geometry", QRect(normal_geometry))
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("window/main_splitter", self.main_splitter.saveState())

    def clamp_window_to_available_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        if frame.width() > available.width() or frame.height() > available.height():
            self.resize(min(self.width(), available.width()), min(self.height(), available.height()))
            frame = self.frameGeometry()

        x = min(max(frame.x(), available.x()), available.right() - frame.width() + 1)
        y = min(max(frame.y(), available.y()), available.bottom() - frame.height() + 1)
        if x != frame.x() or y != frame.y():
            self.move(x, y)

    def center_window_on_available_screen(self) -> None:
        if self.is_window_maximized() or self.isFullScreen():
            return
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def _create_refresh_button(self, object_name: str, tooltip: str) -> QPushButton:
        button = CompactButton()
        button.setObjectName(object_name)
        button.setToolTip(tooltip)
        button.setFixedSize(34, 30)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        icon = RefreshIcon(button)
        icon.setObjectName("refreshIcon")
        icon.setFixedSize(RefreshIcon.ICON_SIZE)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QHBoxLayout(button)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(icon, 1, Qt.AlignmentFlag.AlignCenter)
        return button

    def _build_top_bar(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.input_combo = SearchableComboBox(empty_text=self.t("empty.inputs"), search_text=self.t("common.search"))
        self.output_combo = SearchableComboBox(empty_text=self.t("empty.outputs"), search_text=self.t("common.search"))
        self.sample_rate_combo = SearchableComboBox(empty_text=self.t("empty.sample_rates"), search_text=self.t("common.search"))
        self.audio_dtype_combo = SearchableComboBox(empty_text=self.t("empty.formats"), search_text=self.t("common.search"))
        self._configure_flexible_combo(self.input_combo, min_chars=16)
        self._configure_flexible_combo(self.output_combo, min_chars=16)
        self.sample_rate_combo.setFixedWidth(104)
        self.audio_dtype_combo.setFixedWidth(118)
        self.input_combo.currentIndexChanged.connect(self.refresh_audio_settings)
        self.output_combo.currentIndexChanged.connect(self.refresh_audio_settings)
        self.sample_rate_combo.currentIndexChanged.connect(self.refresh_audio_dtype_options)
        self.refresh_devices_button = self._create_refresh_button("refreshDevicesButton", self.t("tooltip.refresh_devices"))
        self.refresh_devices_button.clicked.connect(lambda: self.refresh_devices(show_feedback=True))

        self.audio_button = CompactButton()
        self.audio_button.setObjectName("audioButton")
        self.audio_button.setProperty("running", False)
        self.audio_button.setToolTip(self.t("audio.start"))
        self.audio_button.setFixedSize(38, 30)
        self.audio_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.audio_button.clicked.connect(self.toggle_audio)
        self.audio_icon_label = AudioTransportIcon(self.audio_button)
        self.audio_icon_label.setObjectName("audioButtonIcon")
        self.audio_icon_label.setProperty("running", False)
        self.audio_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.audio_icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        audio_button_layout = QHBoxLayout(self.audio_button)
        audio_button_layout.setContentsMargins(0, 0, 0, 0)
        audio_button_layout.addWidget(self.audio_icon_label, 1, Qt.AlignmentFlag.AlignCenter)

        self.input_label = QLabel(self.t("top.input"))
        self.output_label = QLabel(self.t("top.output"))
        self.sample_rate_label = QLabel(self.t("top.sample_rate"))
        self.audio_dtype_label = QLabel(self.t("top.format"))

        layout.addWidget(self.input_label, 0, 0)
        layout.addWidget(self.input_combo, 0, 1)
        layout.addWidget(self.output_label, 0, 2)
        layout.addWidget(self.output_combo, 0, 3)
        layout.addWidget(self.sample_rate_label, 0, 4)
        layout.addWidget(self.sample_rate_combo, 0, 5)
        layout.addWidget(self.audio_dtype_label, 0, 6)
        layout.addWidget(self.audio_dtype_combo, 0, 7)
        layout.addWidget(self.refresh_devices_button, 0, 8)
        layout.addWidget(self.audio_button, 0, 9)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(3, 2)
        return panel

    def _build_settings_menu(self) -> None:
        content = RoundedPopupPanel(self)
        content.setObjectName("settingsPanel")
        content.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        content.setFixedWidth(SETTINGS_PANEL_WIDTH)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.settings_audio_title = QLabel(self.t("settings.audio"))
        self.settings_audio_title.setObjectName("settingsTitle")
        layout.addWidget(self.settings_audio_title)

        latency_row = QHBoxLayout()
        self.current_latency_label = QLabel(self.t("settings.current_latency"))
        latency_row.addWidget(self.current_latency_label)
        latency_row.addStretch(1)
        self.status_label = QLabel("--")
        self.status_label.setObjectName("latencyLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        latency_row.addWidget(self.status_label)
        layout.addLayout(latency_row)

        self.custom_latency_check = AieqCheckBox()
        self.set_wrapped_checkbox_text(self.custom_latency_check, self.t("settings.custom_latency"))
        self.custom_latency_check.setChecked(self._settings_bool("audio/custom_latency_enabled", False))
        self.custom_latency_check.toggled.connect(self.on_custom_latency_toggled)
        layout.addWidget(self.custom_latency_check)

        custom_latency_row = QHBoxLayout()
        self.custom_latency_label = QLabel(self.t("settings.latency_ms"))
        custom_latency_row.addWidget(self.custom_latency_label)
        self.custom_latency_spin = NoWheelDoubleSpinBox()
        self.custom_latency_spin.setDecimals(1)
        self.custom_latency_spin.setRange(1.0, 1000.0)
        self.custom_latency_spin.setSingleStep(5.0)
        self.custom_latency_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.custom_latency_spin.setValue(self._settings_float("audio/custom_latency_ms", 50.0))
        self.custom_latency_spin.setEnabled(self.custom_latency_check.isChecked())
        self.custom_latency_spin.setFixedWidth(190)
        self.custom_latency_spin.valueChanged.connect(self.save_audio_settings)
        custom_latency_row.addWidget(self.custom_latency_spin)
        layout.addLayout(custom_latency_row)

        self.settings_ai_title = QLabel(self.t("settings.ai"))
        self.settings_ai_title.setObjectName("settingsTitle")
        layout.addWidget(self.settings_ai_title)

        self.allow_cpu_fallback_check = AieqCheckBox()
        self.set_wrapped_checkbox_text(self.allow_cpu_fallback_check, self.t("settings.allow_cpu"))
        self.allow_cpu_fallback_check.setChecked(self._settings_bool("ai/allow_cpu_fallback", False))
        self.allow_cpu_fallback_check.toggled.connect(self.save_ai_settings)
        layout.addWidget(self.allow_cpu_fallback_check)

        self.advanced_ai_check = AieqCheckBox()
        self.set_wrapped_checkbox_text(self.advanced_ai_check, self.t("settings.advanced_ai"))
        self.advanced_ai_check.setChecked(self._settings_bool("ai/advanced_enabled", False))
        self.advanced_ai_check.toggled.connect(self.on_advanced_ai_toggled)
        layout.addWidget(self.advanced_ai_check)

        ai_grid = QGridLayout()
        ai_grid.setHorizontalSpacing(8)
        ai_grid.setVerticalSpacing(6)
        self.ai_ctx_spin = QSpinBox()
        self.ai_ctx_spin.setRange(512, 131072)
        self.ai_ctx_spin.setSingleStep(512)
        self.ai_ctx_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.ai_ctx_spin.setValue(self._settings_int("ai/n_ctx", self.ai_service.llama_n_ctx))
        self.ai_ctx_spin.valueChanged.connect(self.on_ai_settings_value_changed)
        self.ai_max_tokens_spin = QSpinBox()
        self.ai_max_tokens_spin.setRange(128, 131072)
        self.ai_max_tokens_spin.setSingleStep(128)
        self.ai_max_tokens_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.ai_max_tokens_spin.setValue(self._settings_int("ai/max_tokens", self.ai_service.llama_max_tokens))
        self.ai_max_tokens_spin.valueChanged.connect(self.on_ai_settings_value_changed)
        self.ai_temperature_spin = NoWheelDoubleSpinBox()
        self.ai_temperature_spin.setDecimals(2)
        self.ai_temperature_spin.setRange(0.0, 2.0)
        self.ai_temperature_spin.setSingleStep(0.05)
        self.ai_temperature_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.ai_temperature_spin.setValue(self._settings_float("ai/temperature", self.ai_service.llama_temperature))
        self.ai_temperature_spin.valueChanged.connect(self.on_ai_settings_value_changed)
        self.ai_model_limit_label = QLabel(self.max_hint("--"))
        self.ai_model_limit_label.setObjectName("settingsHint")
        self.ai_tokens_limit_label = QLabel(self.max_hint("--"))
        self.ai_tokens_limit_label.setObjectName("settingsHint")
        self.ai_temperature_limit_label = QLabel(self.max_hint("--"))
        self.ai_temperature_limit_label.setObjectName("settingsHint")
        for label in (self.ai_model_limit_label, self.ai_tokens_limit_label, self.ai_temperature_limit_label):
            label.setFixedWidth(SETTINGS_AI_HINT_WIDTH)

        self.ai_ctx_label = QLabel(self.t("settings.context"))
        self.ai_max_tokens_label = QLabel(self.t("settings.tokens"))
        self.ai_temperature_label = QLabel(self.t("settings.temperature"))
        for label in (self.ai_ctx_label, self.ai_max_tokens_label, self.ai_temperature_label):
            label.setFixedWidth(SETTINGS_AI_LABEL_WIDTH)

        ai_grid.addWidget(self.ai_ctx_label, 0, 0)
        ai_grid.addWidget(self.ai_ctx_spin, 0, 1)
        ai_grid.addWidget(self.ai_model_limit_label, 0, 2)
        ai_grid.addWidget(self.ai_max_tokens_label, 1, 0)
        ai_grid.addWidget(self.ai_max_tokens_spin, 1, 1)
        ai_grid.addWidget(self.ai_tokens_limit_label, 1, 2)
        ai_grid.addWidget(self.ai_temperature_label, 2, 0)
        ai_grid.addWidget(self.ai_temperature_spin, 2, 1)
        ai_grid.addWidget(self.ai_temperature_limit_label, 2, 2)
        layout.addLayout(ai_grid)

        self.settings_app_title = QLabel(self.t("settings.app"))
        self.settings_app_title.setObjectName("settingsTitle")
        layout.addWidget(self.settings_app_title)

        language_row = QHBoxLayout()
        self.language_label = QLabel(self.t("settings.language"))
        language_row.addWidget(self.language_label)
        language_row.addStretch(1)
        self.refresh_languages_button = self._create_refresh_button("refreshLanguagesButton", self.t("tooltip.refresh_languages"))
        self.refresh_languages_button.clicked.connect(lambda: self.refresh_languages(show_feedback=True))
        language_row.addWidget(self.refresh_languages_button)
        self.language_combo = SearchableComboBox(empty_text=self.t("settings.no_languages"), search_text=self.t("common.search"))
        self.language_combo.setFixedWidth(190)
        self.language_combo.currentIndexChanged.connect(self.on_language_changed)
        language_row.addWidget(self.language_combo)
        layout.addLayout(language_row)

        self.settings_panel = content
        self.refresh_language_combo()
        self.on_custom_latency_toggled(self.custom_latency_check.isChecked())
        self.on_advanced_ai_toggled(self.advanced_ai_check.isChecked())
        self.refresh_ai_settings_limits()
        self.resize_settings_panel()

    def show_settings_menu(self) -> None:
        self.refresh_ai_settings_limits()
        self.update_audio_latency_label()
        self.refresh_latency_settings_state()
        panel = self.settings_panel
        self.resize_settings_panel()
        position = self.settings_button.mapToGlobal(QPoint(0, self.settings_button.height() + 6))
        panel.move(position)
        panel.show()
        panel.raise_()

    def open_github_repository(self) -> None:
        QDesktopServices.openUrl(QUrl("https://github.com/dmitryz1024/aieq"))

    def _settings_bool(self, key: str, default: bool) -> bool:
        value = self.settings.value(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().casefold() in {"1", "true", "yes", "on"}

    def _settings_int(self, key: str, default: int) -> int:
        try:
            return int(self.settings.value(key, default))
        except (TypeError, ValueError):
            return default

    def _settings_float(self, key: str, default: float) -> float:
        try:
            return float(self.settings.value(key, default))
        except (TypeError, ValueError):
            return default

    def on_custom_latency_toggled(self, checked: bool) -> None:
        self.refresh_latency_settings_state()
        self.save_audio_settings()

    def refresh_latency_settings_state(self) -> None:
        if not hasattr(self, "custom_latency_check"):
            return
        running = self.audio_engine.is_running
        custom_enabled = self.custom_latency_check.isChecked()
        self.custom_latency_check.setEnabled(not running)
        self.custom_latency_label.setEnabled(not running and custom_enabled)
        self.custom_latency_spin.setEnabled(not running and custom_enabled)

    def save_audio_settings(self, *_args) -> None:
        if not hasattr(self, "custom_latency_check"):
            return
        self.settings.setValue("audio/custom_latency_enabled", self.custom_latency_check.isChecked())
        self.settings.setValue("audio/custom_latency_ms", self.custom_latency_spin.value())

    def selected_audio_latency(self) -> tuple[str | float | tuple[str | float, str | float], bool]:
        if not hasattr(self, "custom_latency_check") or not self.custom_latency_check.isChecked():
            return "low", False
        return ("low", max(0.001, self.custom_latency_spin.value() / 1000.0)), True

    def on_advanced_ai_toggled(self, checked: bool) -> None:
        for widget in (
            self.ai_ctx_label,
            self.ai_ctx_spin,
            self.ai_model_limit_label,
            self.ai_max_tokens_label,
            self.ai_max_tokens_spin,
            self.ai_tokens_limit_label,
            self.ai_temperature_label,
            self.ai_temperature_spin,
            self.ai_temperature_limit_label,
        ):
            widget.setEnabled(checked)
        self.save_ai_settings()

    def on_ai_settings_value_changed(self, _value: int | float) -> None:
        self.refresh_ai_settings_limits()
        self.save_ai_settings()

    def save_ai_settings(self, *_args) -> None:
        if not hasattr(self, "advanced_ai_check"):
            return
        self.settings.setValue("ai/advanced_enabled", self.advanced_ai_check.isChecked())
        self.settings.setValue("ai/allow_cpu_fallback", self.allow_cpu_fallback_check.isChecked())
        self.settings.setValue("ai/n_ctx", self.ai_ctx_spin.value())
        self.settings.setValue("ai/max_tokens", self.ai_max_tokens_spin.value())
        self.settings.setValue("ai/temperature", self.ai_temperature_spin.value())

    def refresh_ai_settings_limits(self, _index: int | None = None) -> None:
        if not hasattr(self, "ai_ctx_spin"):
            return
        model_limit = self.selected_model_context_limit()
        ctx_max = model_limit or 131072
        self.ai_ctx_spin.blockSignals(True)
        self.ai_max_tokens_spin.blockSignals(True)
        self.ai_temperature_spin.blockSignals(True)
        current_ctx = min(max(self.ai_ctx_spin.value(), self.ai_ctx_spin.minimum()), ctx_max)
        self.ai_ctx_spin.setMaximum(ctx_max)
        self.ai_ctx_spin.setValue(current_ctx)
        token_max = max(128, min(current_ctx, ctx_max))
        self.ai_max_tokens_spin.setMaximum(token_max)
        if self.ai_max_tokens_spin.value() > token_max:
            self.ai_max_tokens_spin.setValue(token_max)
        temperature_max = self.selected_model_temperature_max()
        self.ai_temperature_spin.setMaximum(temperature_max)
        if self.ai_temperature_spin.value() > temperature_max:
            self.ai_temperature_spin.setValue(temperature_max)
        self.ai_ctx_spin.blockSignals(False)
        self.ai_max_tokens_spin.blockSignals(False)
        self.ai_temperature_spin.blockSignals(False)
        self.ai_model_limit_label.setText(self.max_hint(str(model_limit) if model_limit else "--"))
        self.ai_tokens_limit_label.setText(self.max_hint(str(token_max)))
        self.ai_temperature_limit_label.setText(self.max_hint(f"{temperature_max:.2f}"))

    def max_hint(self, value: str) -> str:
        return f"{self.t('settings.max_prefix')}: {value}"

    def selected_model_context_limit(self) -> int | None:
        model_path = self.selected_ai_model_path() if hasattr(self, "ai_model_combo") else None
        if model_path is None:
            return None
        key = str(model_path)
        if key not in self.model_context_limits:
            self.model_context_limits[key] = read_gguf_context_length(model_path)
        return self.model_context_limits[key]

    def selected_model_temperature_max(self) -> float:
        model_path = self.selected_ai_model_path() if hasattr(self, "ai_model_combo") else None
        if model_path is None:
            return 2.0
        name = model_path.name.casefold()
        if "qwen" in name:
            return 2.0
        if "mistral" in name or "mixtral" in name:
            return 1.5
        return 2.0

    def apply_ai_runtime_settings(self) -> None:
        allow_cpu_fallback = (
            self.allow_cpu_fallback_check.isChecked()
            if hasattr(self, "allow_cpu_fallback_check")
            else False
        )
        if not hasattr(self, "advanced_ai_check") or not self.advanced_ai_check.isChecked():
            if self._ai_runtime_signature is not None:
                self.ai_service.shutdown()
            self._ai_runtime_signature = None
            self.ai_service.set_runtime_overrides(allow_cpu_fallback=allow_cpu_fallback)
            return
        signature = (
            self.ai_ctx_spin.value(),
            self.ai_max_tokens_spin.value(),
            round(self.ai_temperature_spin.value(), 3),
        )
        if signature != self._ai_runtime_signature:
            self.ai_service.shutdown()
        self._ai_runtime_signature = signature
        self.ai_service.set_runtime_overrides(
            n_ctx=signature[0],
            max_tokens=signature[1],
            temperature=signature[2],
            allow_cpu_fallback=allow_cpu_fallback,
        )

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_graph_section(), 3)
        layout.addWidget(self._build_filters_section(), 2)
        return panel

    def _build_graph_section(self) -> QGroupBox:
        self.graph_group = QGroupBox(self.t("section.graph"))
        box = self.graph_group
        layout = QVBoxLayout(box)

        pg.setConfigOptions(antialias=True)
        self.plot = RoundedPlotWidget(
            axisItems={
                "bottom": FrequencyAxisItem(orientation="bottom", label_offset=(0.0, -4.0)),
                "left": LabelPaddedAxisItem(orientation="left", label_offset=(4.0, 0.0)),
            }
        )
        self.plot.setObjectName("plotWidget")
        self.plot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.plot.setBackground("#111318")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLogMode(x=True, y=False)
        self.plot.setLabel("bottom", self.t("graph.frequency"))
        self.plot.setLabel("left", self.t("graph.gain"))
        self.plot.getAxis("bottom").setStyle(tickTextOffset=9)
        self.plot.getAxis("left").setStyle(tickTextOffset=9)
        self.plot.setXRange(np.log10(20.0), np.log10(20000.0), padding=0)
        self.plot.setYRange(-20.0, 20.0, padding=0)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.getViewBox().setLimits(
            xMin=np.log10(20.0),
            xMax=np.log10(20000.0),
            yMin=-20.0,
            yMax=20.0,
            minXRange=np.log10(20000.0) - np.log10(20.0),
            maxXRange=np.log10(20000.0) - np.log10(20.0),
            minYRange=40.0,
            maxYRange=40.0,
        )
        plot_item = self.plot.getPlotItem()
        self.plot_legend = HoverLegendItem(
            offset=(-12, 12),
            colCount=1,
            horSpacing=12,
            verSpacing=4,
            brush=pg.mkBrush(17, 19, 24, 210),
            pen=pg.mkPen("#3a414d"),
            labelTextColor="#e8edf2",
            sampleType=AieqLegendSample,
        )
        self.plot_legend.setParentItem(plot_item.vb)
        plot_item.legend = self.plot_legend
        self.device_curve_item = self.plot.plot(
            GRAPH_FREQS,
            np.zeros_like(GRAPH_FREQS),
            pen=pg.mkPen(DEVICE_CURVE_COLOR, width=2),
            name=self.t("graph.device"),
        )
        self.device_curve_item.setZValue(-10)
        self.target_curve_item = self.plot.plot(
            GRAPH_FREQS,
            np.zeros_like(GRAPH_FREQS),
            pen=pg.mkPen(TARGET_COLOR, width=2),
        )
        self.target_curve_item.setZValue(-6)
        self.target_curve_item.hide()
        self.target_legend_visible = False
        self.current_curve = self.plot.plot(
            GRAPH_FREQS,
            np.zeros_like(GRAPH_FREQS),
            pen=pg.mkPen(CURRENT_COLOR, width=3),
            name=self.t("graph.current"),
        )
        self.current_curve.setZValue(10)
        self.compare_curves: dict[int, pg.PlotDataItem] = {}
        layout.addWidget(self.plot, 1)

        controls = QHBoxLayout()
        self.device_curve_combo = SearchableComboBox(empty_text=self.t("empty.devices"), search_text=self.t("common.search"))
        self.device_curve_combo.setProperty("popupVisibleRows", 6)
        self.device_curve_combo.setProperty("popupRowHeight", POPUP_OPTION_DEFAULT_ROW_HEIGHT)
        self._configure_flexible_combo(self.device_curve_combo, min_chars=8)
        self.device_curve_combo.setMaxVisibleItems(12)
        self.device_curve_combo.currentIndexChanged.connect(self.on_device_curve_changed)
        self.refresh_curves_button = self._create_refresh_button("refreshCurvesButton", self.t("tooltip.refresh_lists"))
        self.refresh_curves_button.clicked.connect(lambda: self.refresh_curve_lists(show_feedback=True))
        self.current_selector = SearchableComboBox(empty_text=self.t("empty.presets"), search_text=self.t("common.search"))
        self.current_selector.setProperty("popupVisibleRows", 6)
        self.current_selector.setProperty("popupRowHeight", POPUP_OPTION_DEFAULT_ROW_HEIGHT)
        self._configure_flexible_combo(self.current_selector, min_chars=8)
        self.current_selector.currentIndexChanged.connect(self.load_current_from_selector)
        self.compare_button = CompactButton(self.t("button.compare"))
        self.compare_button.setObjectName("compareButton")
        self.compare_button.setFixedWidth(98)
        self.compare_button.setFixedHeight(30)
        self.compare_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.compare_button.setText(self.t("button.compare"))
        self.compare_menu = QMenu(self.compare_button)
        self.compare_button.clicked.connect(self.show_compare_menu)
        self.import_button = CompactButton(self.t("button.import"))
        self.import_button.setObjectName("miniButton")
        self.import_button.setFixedWidth(82)
        self.import_button.setFixedHeight(30)
        self.import_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.import_button.clicked.connect(self.import_preset)
        self.export_button = CompactButton(self.t("button.export"))
        self.export_button.setObjectName("miniButton")
        self.export_button.setFixedWidth(82)
        self.export_button.setFixedHeight(30)
        self.export_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_button.clicked.connect(self.export_preset)

        self.device_curve_label = CompactLabel(self.t("label.device"), hint_width=86)
        self.device_curve_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.device_curve_label)
        controls.addWidget(self.device_curve_combo, 1)
        controls.addWidget(self.refresh_curves_button)
        self.current_preset_label = CompactLabel(self.t("label.current_preset"), hint_width=118)
        self.current_preset_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.current_preset_label)
        controls.addWidget(self.current_selector, 1)
        controls.addWidget(self.compare_button)
        controls.addWidget(self.import_button)
        controls.addWidget(self.export_button)
        layout.addLayout(controls)
        return box

    def _configure_flexible_combo(self, combo: QComboBox, *, min_chars: int) -> None:
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(min_chars)
        combo.setMinimumWidth(0)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _build_filters_section(self) -> QGroupBox:
        self.filters_group = QGroupBox(self.t("section.filters"))
        box = self.filters_group
        box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(box)

        self.filter_scroll = QScrollArea()
        self.filter_scroll.setObjectName("filterScroll")
        self.filter_scroll.setWidgetResizable(False)
        self.filter_scroll.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.filter_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.filter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.filter_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.filter_scroll.setFixedHeight(FilterEditorRow.FIXED_HEIGHT + 6)
        self.filter_scroll.setMinimumWidth(0)
        self.filter_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.filter_scroll.horizontalScrollBar().setSingleStep(76)
        self.filter_scroll.viewport().setObjectName("filterViewport")
        self.filter_scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.filter_scroll.viewport().installEventFilter(self)
        self.filter_scroll.horizontalScrollBar().rangeChanged.connect(self.schedule_filter_container_sync)
        self.filter_container = QWidget()
        self.filter_container.setObjectName("filterContainer")
        self.filter_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.filter_container.installEventFilter(self)
        self.filter_container.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.filter_list_layout = QHBoxLayout(self.filter_container)
        self.filter_list_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_list_layout.setSpacing(8)
        self.filter_list_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.filter_list_layout.addStretch(1)
        self.filter_scroll.setWidget(self.filter_container)
        self.filter_scroll_frame = QFrame()
        self.filter_scroll_frame.setObjectName("filterScrollFrame")
        self.filter_scroll_frame.setFixedHeight(FilterEditorRow.FIXED_HEIGHT + 12)
        self.filter_scroll_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        filter_frame_layout = QVBoxLayout(self.filter_scroll_frame)
        filter_frame_layout.setContentsMargins(1, 1, 1, 1)
        filter_frame_layout.setSpacing(0)
        filter_frame_layout.addWidget(self.filter_scroll)
        layout.addWidget(self.filter_scroll_frame, 1)

        buttons = QHBoxLayout()
        self.add_filter_button = QPushButton(self.t("button.add"))
        self.add_filter_button.clicked.connect(self.add_filter)
        self.remove_filter_button = QPushButton(self.t("button.remove"))
        self.remove_filter_button.clicked.connect(self.remove_selected_filter)
        self.clear_filters_button = QPushButton(self.t("button.clear"))
        self.clear_filters_button.clicked.connect(self.clear_filters)
        self.delete_preset_button = QPushButton(self.t("button.delete_preset"))
        self.delete_preset_button.clicked.connect(self.delete_current_preset)
        self.save_button = QPushButton(self.t("button.save_preset"))
        self.save_button.clicked.connect(self.save_current_preset)
        buttons.addWidget(self.add_filter_button)
        buttons.addWidget(self.remove_filter_button)
        buttons.addWidget(self.clear_filters_button)
        buttons.addStretch(1)
        buttons.addWidget(self.delete_preset_button)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        return box

    def _build_chat_panel(self) -> QGroupBox:
        self.assistant_group = QGroupBox(self.t("section.assistant"))
        box = self.assistant_group
        layout = QVBoxLayout(box)
        self.side_tabs = QTabWidget()

        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        ai_layout.setContentsMargins(0, 10, 0, 0)
        ai_layout.setSpacing(8)
        model_controls = QHBoxLayout()
        self.ai_model_label = QLabel(self.t("label.model"))
        model_controls.addWidget(self.ai_model_label)
        self.ai_model_combo = SearchableComboBox(empty_text=self.t("empty.models"), search_text=self.t("common.search"))
        self.ai_model_combo.setMaxVisibleItems(12)
        self._configure_flexible_combo(self.ai_model_combo, min_chars=12)
        self.ai_model_combo.currentIndexChanged.connect(self.refresh_ai_settings_limits)
        model_controls.addWidget(self.ai_model_combo, 1)
        self.refresh_models_button = self._create_refresh_button("refreshModelsButton", self.t("tooltip.refresh_models"))
        self.refresh_models_button.clicked.connect(lambda: self.refresh_ai_models(show_feedback=True))
        model_controls.addWidget(self.refresh_models_button)
        self.chat_history = QTextBrowser()
        self.chat_history.setOpenExternalLinks(True)
        self.chat_composer = QFrame()
        self.chat_composer.setObjectName("chatComposer")
        composer_layout = QVBoxLayout(self.chat_composer)
        composer_layout.setContentsMargins(4, 4, 4, 4)
        composer_layout.setSpacing(2)
        self.chat_input = ChatInput()
        self.chat_input.setObjectName("chatInput")
        self.chat_input.setPlaceholderText(self.t("chat.placeholder"))
        self.chat_input.setFixedHeight(76)
        self.chat_input.submit_requested.connect(self.send_chat)
        self.chat_menu_button = QPushButton("≡")
        self.chat_menu_button.setObjectName("chatIconButton")
        self.chat_menu_button.setToolTip(self.t("tooltip.saved_chats"))
        self.chat_menu_button.setFixedSize(30, 30)
        self.chat_menu = QMenu(self.chat_menu_button)
        self.chat_menu_button.clicked.connect(self.show_chat_menu)
        self.delete_chat_button = QPushButton("×")
        self.delete_chat_button.setObjectName("chatIconButton")
        self.delete_chat_button.setToolTip(self.t("tooltip.delete_chat"))
        self.delete_chat_button.setFixedSize(30, 30)
        self.delete_chat_button.clicked.connect(self.delete_current_chat)
        self.new_chat_button = QPushButton("+")
        self.new_chat_button.setObjectName("chatIconButton")
        self.new_chat_button.setToolTip(self.t("tooltip.new_chat"))
        self.new_chat_button.setFixedSize(30, 30)
        self.new_chat_button.clicked.connect(self.start_new_chat)
        self.send_button = QPushButton("↑")
        self.send_button.setObjectName("chatIconButton")
        self.send_button.setToolTip(self.t("tooltip.send"))
        self.send_button.setFixedSize(30, 30)
        self.send_button.clicked.connect(self.send_chat)
        chat_buttons = QHBoxLayout()
        chat_buttons.addWidget(self.chat_menu_button)
        chat_buttons.addWidget(self.delete_chat_button)
        chat_buttons.addStretch(1)
        chat_buttons.addWidget(self.new_chat_button)
        chat_buttons.addWidget(self.send_button)
        composer_layout.addWidget(self.chat_input)
        composer_layout.addLayout(chat_buttons)
        ai_layout.addLayout(model_controls)
        ai_layout.addWidget(self.chat_history, 1)
        ai_layout.addWidget(self.chat_composer)
        self.append_chat("AIEQ", self.t("chat.intro"))

        autoeq_tab = QWidget()
        autoeq_layout = QVBoxLayout(autoeq_tab)
        autoeq_layout.setContentsMargins(0, 10, 0, 0)
        autoeq_layout.setSpacing(8)
        self.target_curve_label = QLabel(self.t("label.target_curve"))
        autoeq_layout.addWidget(self.target_curve_label)
        self.target_curve_combo = SearchableComboBox(empty_text=self.t("empty.curves"), search_text=self.t("common.search"))
        self._configure_flexible_combo(self.target_curve_combo, min_chars=12)
        self.target_curve_combo.setMaxVisibleItems(12)
        self.target_curve_combo.currentIndexChanged.connect(self.on_target_curve_changed)
        autoeq_layout.addWidget(self.target_curve_combo)
        self.show_target_checkbox = AieqCheckBox(self.t("checkbox.show_target"))
        self.show_target_checkbox.toggled.connect(lambda _checked: self.update_graph())
        autoeq_layout.addWidget(self.show_target_checkbox)
        self.autoeq_algorithm_label = QLabel(self.t("label.algorithm"))
        autoeq_layout.addWidget(self.autoeq_algorithm_label)
        self.autoeq_backend_combo = SimpleComboBox()
        self.autoeq_backend_combo.setProperty("disablePopupWheel", True)
        self.autoeq_backend_combo.setProperty("popupVisibleRows", 2)
        self.autoeq_backend_combo.setProperty("popupRowHeight", POPUP_OPTION_DEFAULT_ROW_HEIGHT)
        self.autoeq_backend_combo.addItem("dmitryz1024", "local")
        self.autoeq_backend_combo.addItem("jaakkopasanen", "official")
        autoeq_layout.addWidget(self.autoeq_backend_combo)
        self.run_autoeq_button = QPushButton(self.t("button.run_autoeq"))
        self.run_autoeq_button.clicked.connect(self.run_autoeq)
        autoeq_layout.addWidget(self.run_autoeq_button)
        autoeq_layout.addStretch(1)

        self.side_tabs.addTab(ai_tab, self.t("tab.ai_chat"))
        self.side_tabs.addTab(autoeq_tab, self.t("tab.autoeq"))
        layout.addWidget(self.side_tabs, 1)
        return box

    @staticmethod
    def _apply_style() -> None:
        QApplication.instance().setStyleSheet(
            """
            QWidget {
                background: #181a1f;
                color: #e8edf2;
                font-size: 13px;
            }
            QWidget#windowRoot {
                background: transparent;
            }
            QFrame#windowFrame {
                background: #181a1f;
                border: 1px solid #313640;
                border-radius: 10px;
            }
            QFrame#windowFrame[maximized="true"] {
                border-radius: 0;
                border: 0;
            }
            QFrame#dialogFrame {
                background: #181a1f;
                border: 1px solid #313640;
                border-radius: 10px;
            }
            QWidget#dialogBody {
                background: transparent;
            }
            QLabel#dialogTitle {
                color: #f0f3f6;
                font-weight: 600;
                background: transparent;
            }
            QLabel#dialogMessage,
            QLabel#dialogInputLabel {
                color: #dce3ea;
                background: transparent;
            }
            QLineEdit#dialogInputEdit {
                background: #111318;
                border: 1px solid #313640;
                border-radius: 6px;
                color: #cfd6df;
                padding: 5px;
                selection-background-color: #05e5b6;
                selection-color: #111318;
                outline: none;
            }
            QLineEdit#dialogInputEdit:focus {
                border-color: #586171;
                outline: none;
            }
            QWidget#contentRoot {
                background: transparent;
            }
            QWidget#windowTitleBar {
                background: #181a1f;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
            QLabel#windowTitleLogo {
                background: transparent;
            }
            QWidget#windowControls {
                background: transparent;
            }
            QPushButton#windowControlButton, QPushButton#windowCloseButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                color: #cfd6df;
                font-size: 15px;
                font-weight: 600;
                padding: 0;
            }
            QPushButton#windowControlButton:hover {
                background: #242933;
                border-color: #3a414d;
                color: #ffffff;
            }
            QPushButton#windowCloseButton:hover {
                background: #532126;
                border-color: #d44444;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #313640;
                border-radius: 8px;
                margin-top: 24px;
                padding: 14px 10px 10px 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 7px;
                color: #f0f3f6;
            }
            QPushButton {
                background: #242933;
                border: 1px solid #3a414d;
                border-radius: 6px;
                padding: 7px 11px;
                outline: none;
            }
            QPushButton:focus, QComboBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
                outline: none;
            }
            QPushButton:hover {
                background: #2d3440;
                border-color: #586171;
            }
            QPushButton:pressed {
                background: #1f242c;
            }
            QPushButton[defaultDialogButton="true"] {
                border-color: #05e5b6;
            }
            QPushButton[defaultDialogButton="true"]:hover {
                background: #2d3440;
                border-color: #05e5b6;
            }
            QPushButton[defaultDialogButton="true"]:pressed {
                background: #1f242c;
                border-color: #05e5b6;
            }
            QPushButton:disabled {
                background: #1d222b;
                border-color: #2b313b;
                color: #707987;
            }
            QPushButton#audioButton {
                padding: 0;
                text-align: center;
            }
            QLabel#audioButtonIcon {
                background: transparent;
                color: #ffffff;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#audioButtonIcon[running="true"] {
                font-size: 11px;
            }
            QPushButton#audioButton[running="false"] {
                background: #025443;
                border-color: #05e5b6;
                color: #ffffff;
            }
            QPushButton#audioButton[running="false"]:hover {
                background: #036955;
                border-color: #35f0c8;
            }
            QPushButton#audioButton[running="true"] {
                background: #532126;
                border-color: #d44444;
                color: #ffffff;
            }
            QPushButton#audioButton[running="true"]:hover {
                background: #6a2930;
                border-color: #e35a5a;
            }
            QPushButton#refreshDevicesButton, QPushButton#refreshCurvesButton, QPushButton#refreshModelsButton, QPushButton#refreshLanguagesButton {
                padding: 0;
            }
            QLabel#refreshIcon {
                background: transparent;
            }
            QPushButton#refreshDevicesButton:disabled {
                background: #1b2028;
                border-color: #272e38;
                color: #626b78;
            }
            QPushButton#miniButton, QPushButton#compareButton {
                padding: 4px 8px;
                font-size: 12px;
                min-height: 20px;
                max-height: 30px;
            }
            QPushButton#chatIconButton {
                background: transparent;
                border: 0;
                color: #9aa4b2;
                font-size: 20px;
                font-weight: 600;
                padding: 0;
            }
            QPushButton#chatIconButton:hover {
                color: #f5fbff;
                background: transparent;
            }
            QPushButton#chatIconButton:disabled {
                color: #4e5664;
                background: transparent;
            }
            QPushButton#settingsIconButton {
                background: transparent;
                border: 0;
                color: #9aa4b2;
                font-size: 16px;
                font-weight: 600;
                padding: 0;
            }
            QPushButton#settingsIconButton:hover {
                color: #f5fbff;
                background: transparent;
            }
            QPushButton#inlineIconButton {
                background: transparent;
                border: 0;
                padding: 0;
            }
            QPushButton#inlineIconButton:hover {
                background: transparent;
            }
            QComboBox, QTextEdit, QTextBrowser, QScrollArea, QDoubleSpinBox, QSpinBox, QTableWidget {
                background: #111318;
                border: 1px solid #313640;
                border-radius: 6px;
                padding: 5px;
                selection-background-color: #05e5b6;
                selection-color: #111318;
                outline: none;
            }
            QComboBox {
                padding-right: 5px;
            }
            QComboBox:disabled {
                color: #69727f;
                background: #101218;
                border-color: #252b35;
            }
            QComboBox::drop-down {
                width: 0px;
                border: 0px;
                background: transparent;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QDoubleSpinBox, QSpinBox {
                padding: 5px;
                selection-background-color: #05e5b6;
                selection-color: #111318;
            }
            QLineEdit#hyperparamEdit {
                background: #111318;
                border: 1px solid #313640;
                border-radius: 6px;
                padding: 5px;
                selection-background-color: #05e5b6;
                selection-color: #111318;
                outline: none;
            }
            QLineEdit#hyperparamEdit:focus {
                outline: none;
                border-color: #586171;
            }
            QScrollArea#hyperparamScroll,
            QWidget#hyperparamContent,
            QFrame#trainingTabPage {
                background: transparent;
                border: 0;
            }
            QFrame#trainingWorkflowFrame {
                background: transparent;
                border: 1px solid #313640;
                border-radius: 8px;
            }
            QGraphicsView#plotWidget {
                background: #111318;
                border: 0;
                border-radius: 7px;
            }
            QDoubleSpinBox:disabled, QSpinBox:disabled {
                color: #69727f;
                background: #101218;
                border-color: #252b35;
            }
            QLabel:disabled {
                color: #69727f;
            }
            QFrame#chatComposer {
                background: #111318;
                border: 1px solid #313640;
                border-radius: 6px;
            }
            QTextEdit#chatInput {
                background: transparent;
                border: 0;
                padding: 5px;
                selection-background-color: #05e5b6;
                selection-color: #111318;
            }
            QTextEdit#readonlyContext {
                color: #69727f;
                background: #101218;
                border-color: #252b35;
            }
            QTextEdit#trainingLog {
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
            }
            QTableWidget#metricsTable {
                gridline-color: #313640;
                selection-background-color: #05e5b6;
                selection-color: #111318;
            }
            QHeaderView::section {
                background: #202632;
                color: #dce3ea;
                border: 0;
                padding: 5px;
                font-weight: 500;
            }
            QScrollBar:horizontal {
                height: 0px;
                margin: 0px;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 0px;
                margin: 0px;
                background: transparent;
            }
            QScrollBar::handle:horizontal,
            QScrollBar::handle:vertical,
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                width: 0px;
                height: 0px;
                background: transparent;
                border: 0px;
            }
            QFrame#filterScrollFrame {
                background: #181a1f;
                border: 1px solid #313640;
                border-radius: 6px;
                padding: 0;
            }
            QScrollArea#filterScroll {
                background: transparent;
                border: 0;
                border-radius: 0;
                padding: 0;
            }
            QWidget#filterViewport,
            QWidget#filterContainer {
                background: #181a1f;
            }
            QCheckBox {
                background: transparent;
                spacing: 8px;
            }
            QCheckBox:disabled {
                color: #69727f;
            }
            QTabWidget::pane {
                border: 0;
            }
            QTabBar::tab {
                background: #242933;
                border: 1px solid #3a414d;
                border-bottom: 0;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 7px 12px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: #111318;
                color: #ffffff;
            }
            QTabBar#trainingTitleTabs::tab {
                border-bottom: 1px solid #3a414d;
                border-radius: 6px;
                padding: 2px 14px;
                min-height: 14px;
                margin-right: 4px;
            }
            QTabBar#trainingTitleTabs::tab:selected {
                border-color: #586171;
            }
            QFrame#filterRow {
                background: #141720;
                border: 1px solid #303745;
                border-radius: 8px;
            }
            QFrame#filterRow[selected="true"] {
                border: 1px solid #05e5b6;
                background: #1d2029;
            }
            QLabel#paramLabel {
                background: #202632;
                border: 1px solid #343c4b;
                border-radius: 5px;
                color: #dce3ea;
                font-size: 13px;
                padding: 2px 6px;
                font-weight: 500;
            }
            QLabel#paramLabel[hovered="true"] {
                background: #2d3440;
                border-color: #586171;
                color: #ffffff;
            }
            QDial {
                background: transparent;
                outline: none;
            }
            QLabel {
                color: #cfd6df;
            }
            QLabel#toast {
                background: rgba(20, 23, 32, 225);
                color: #f0f3f6;
                border: 1px solid #495263;
                border-radius: 8px;
                padding: 8px 14px;
            }
            QMenu {
                background: #181a1f;
                border: 1px solid #3a414d;
            }
            QMenu::item {
                padding: 6px 18px;
            }
            QMenu::item:selected {
                background: #2d3440;
            }
            QMenu#searchPopup {
                padding: 0;
                border: 0;
                background: transparent;
            }
            QWidget#searchPopupPanel {
                background: transparent;
                border: 0;
            }
            QMenu#settingsMenu {
                padding: 0;
            }
            QWidget#settingsPanel {
                background: #181a1f;
                min-width: 330px;
            }
            QLabel#settingsTitle {
                color: #f0f3f6;
                font-weight: 600;
            }
            QLabel#settingsHint {
                color: #8f99a8;
            }
            QLabel#filterTypeInfoImage {
                background: #111318;
                border: 1px solid #313640;
                border-radius: 6px;
                padding: 6px;
            }
            QLabel#filterTypeInfoTitle {
                color: #f0f3f6;
                font-size: 14px;
                font-weight: 600;
                background: transparent;
            }
            QWidget#filterTypeInfoText {
                color: #cfd6df;
                background: transparent;
            }
            QLineEdit#comboSearch, QListWidget#comboSearchList, QListWidget#compareSearchList {
                background: #111318;
                border: 1px solid #313640;
                border-radius: 6px;
                padding: 5px;
                selection-background-color: #05e5b6;
                selection-color: #111318;
                outline: none;
            }
            QListWidget#comboSearchList::item {
                padding: 0;
            }
            QListWidget#compareSearchList::item {
                padding: 0;
            }
            QListWidget#comboSearchList::item:hover {
                background: transparent;
            }
            QListWidget#compareSearchList::item:hover,
            QListWidget#comboSearchList::item:selected,
            QListWidget#compareSearchList::item:selected {
                background: transparent;
            }
            QListWidget#comboSearchList::item:disabled,
            QListWidget#compareSearchList::item:disabled {
                color: #707987;
            }
            """
        )

    def refresh_devices(self, *, show_feedback: bool = False) -> None:
        if self.audio_engine.is_running:
            self.refresh_devices_button.setEnabled(False)
            return
        self.refresh_devices_button.setEnabled(True)
        previous_input = self.input_combo.currentData()
        previous_output = self.output_combo.currentData()
        previous_input_signature = self._current_audio_device_signature(self.input_combo, self.input_devices)
        previous_output_signature = self._current_audio_device_signature(self.output_combo, self.output_devices)
        self.input_combo.blockSignals(True)
        self.output_combo.blockSignals(True)
        self.input_combo.clear()
        self.output_combo.clear()
        try:
            if not self.audio_engine.is_running:
                refresh_audio_backend()
            self.input_devices = list_audio_devices("input")
            self.output_devices = list_audio_devices("output")
        except Exception as exc:  # noqa: BLE001
            self.input_combo.blockSignals(False)
            self.output_combo.blockSignals(False)
            self.status_label.setToolTip(str(exc))
            self.update_audio_latency_label()
            self.show_toast(self.t("toast.sounddevice_unavailable"))
            self.audio_button.setEnabled(False)
            self.refresh_devices_button.setEnabled(True)
            return

        for device in self.input_devices:
            self.input_combo.addItem(self._audio_device_display_label(device, self.input_devices), device.index)
        for device in self.output_devices:
            self.output_combo.addItem(self._audio_device_display_label(device, self.output_devices), device.index)

        self._restore_audio_device_combo(self.input_combo, self.input_devices, previous_input, previous_input_signature)
        self._restore_audio_device_combo(self.output_combo, self.output_devices, previous_output, previous_output_signature)
        self.input_combo.blockSignals(False)
        self.output_combo.blockSignals(False)
        self.refresh_audio_settings()

        self.audio_button.setEnabled(bool(self.input_devices and self.output_devices))
        self.refresh_devices_button.setEnabled(True)
        self.status_label.setToolTip("")
        self.update_audio_latency_label()
        if show_feedback:
            self.show_toast(self.t("toast.driver_lists_updated"))

    def _restore_combo_data(self, combo: QComboBox, value: object) -> None:
        if value is None:
            return
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _audio_device_signature(device: AudioDevice) -> tuple[str, str]:
        return (device.hostapi.casefold(), device.name.casefold())

    @staticmethod
    def _audio_device_display_label(device: AudioDevice, devices: list[AudioDevice]) -> str:
        name = device.name
        if "mme" in device.hostapi.casefold():
            prefix = name.casefold()
            candidates = [
                candidate.name
                for candidate in devices
                if candidate.index != device.index
                and len(candidate.name) > len(name)
                and candidate.name.casefold().startswith(prefix)
            ]
            if candidates:
                name = max(candidates, key=len)
        return f"[{device.hostapi}] {name}"

    def _current_audio_device_signature(self, combo: QComboBox, devices: list[AudioDevice]) -> tuple[str, str] | None:
        selected_index = combo.currentData()
        for device in devices:
            if device.index == selected_index:
                return self._audio_device_signature(device)
        return None

    def _restore_audio_device_combo(
        self,
        combo: QComboBox,
        devices: list[AudioDevice],
        previous_index: object,
        previous_signature: tuple[str, str] | None,
    ) -> None:
        if previous_signature is not None:
            for row, device in enumerate(devices):
                if self._audio_device_signature(device) == previous_signature:
                    combo.setCurrentIndex(row)
                    return
        self._restore_combo_data(combo, previous_index)

    def refresh_audio_settings(self, _index: int | None = None) -> None:
        if not hasattr(self, "sample_rate_combo"):
            return
        previous_rate = self.sample_rate_combo.currentData()
        previous_dtype = self.audio_dtype_combo.currentData()
        input_device = self._selected_device(self.input_combo, self.input_devices)
        output_device = self._selected_device(self.output_combo, self.output_devices)
        self.sample_rate_combo.blockSignals(True)
        self.audio_dtype_combo.blockSignals(True)
        self.sample_rate_combo.clear()
        self.audio_dtype_combo.clear()
        self.audio_settings = []
        if input_device is None or output_device is None:
            self.sample_rate_combo.setEnabled(False)
            self.audio_dtype_combo.setEnabled(False)
            self.sample_rate_combo.blockSignals(False)
            self.audio_dtype_combo.blockSignals(False)
            return
        try:
            self.audio_settings = list_supported_stream_settings(input_device, output_device)
        except Exception:  # noqa: BLE001 - start will surface the real audio error if needed.
            fallback_rate = int(output_device.default_samplerate or input_device.default_samplerate or DEFAULT_SAMPLE_RATE)
            self.audio_settings = [AudioStreamSetting(fallback_rate, "float32")]

        rates = sorted({setting.sample_rate for setting in self.audio_settings})
        for rate in rates:
            self.sample_rate_combo.addItem(str(rate), rate)
        wanted_rate = previous_rate if previous_rate in rates else int(output_device.default_samplerate or rates[0])
        self._restore_combo_data(self.sample_rate_combo, wanted_rate)
        if self.sample_rate_combo.currentIndex() < 0 and self.sample_rate_combo.count():
            self.sample_rate_combo.setCurrentIndex(0)
        self.sample_rate_combo.blockSignals(False)
        self.audio_dtype_combo.blockSignals(False)
        self.refresh_audio_dtype_options(previous_dtype=previous_dtype)
        enabled = not self.audio_engine.is_running
        self.sample_rate_combo.setEnabled(enabled and self.sample_rate_combo.count() > 0)
        self.audio_dtype_combo.setEnabled(enabled and self.audio_dtype_combo.count() > 0)

    def refresh_audio_dtype_options(self, _index: int | None = None, *, previous_dtype: object | None = None) -> None:
        if not hasattr(self, "audio_dtype_combo"):
            return
        if previous_dtype is None:
            previous_dtype = self.audio_dtype_combo.currentData()
        selected_rate = self.sample_rate_combo.currentData()
        self.audio_dtype_combo.blockSignals(True)
        self.audio_dtype_combo.clear()
        for setting in self.audio_settings:
            if setting.sample_rate == selected_rate:
                self.audio_dtype_combo.addItem(setting.dtype_label, setting.dtype)
        self._restore_combo_data(self.audio_dtype_combo, previous_dtype)
        if self.audio_dtype_combo.currentIndex() < 0 and self.audio_dtype_combo.count():
            self.audio_dtype_combo.setCurrentIndex(0)
        self.audio_dtype_combo.blockSignals(False)

    def selected_sample_rate(self) -> int:
        value = self.sample_rate_combo.currentData() if hasattr(self, "sample_rate_combo") else None
        return int(value or DEFAULT_SAMPLE_RATE)

    def selected_audio_dtype(self) -> str:
        value = self.audio_dtype_combo.currentData() if hasattr(self, "audio_dtype_combo") else None
        return str(value or "float32")

    def refresh_ai_models(self, *, show_feedback: bool = False) -> None:
        if not hasattr(self, "ai_model_combo"):
            return
        previous = self.ai_model_combo.currentData()
        default_path = str(self.ai_service.llama_model_path)
        self.ai_model_combo.blockSignals(True)
        self.ai_model_combo.clear()
        models = list_local_models()
        if not models:
            self.ai_model_combo.addItem(self.t("empty.models_not_found"), None)
            self.ai_model_combo.setEnabled(False)
            self.ai_model_combo.blockSignals(False)
            self.refresh_ai_settings_limits()
            if show_feedback:
                self.show_toast(self.t("toast.model_lists_updated"))
            return
        self.ai_model_combo.setEnabled(True)
        selected_index = 0
        wanted = str(previous) if previous else default_path
        for index, model_path in enumerate(models):
            resolved = str(model_path)
            display_name = model_path.stem if model_path.suffix.casefold() == ".gguf" else model_path.name
            self.ai_model_combo.addItem(display_name, resolved)
            if resolved == wanted or str(model_path.resolve()) == wanted:
                selected_index = index
        self.ai_model_combo.setCurrentIndex(selected_index)
        self.ai_model_combo.blockSignals(False)
        self.refresh_ai_settings_limits()
        if show_feedback:
            self.show_toast(self.t("toast.model_lists_updated"))

    def selected_ai_model_path(self) -> Path | None:
        if not hasattr(self, "ai_model_combo"):
            return None
        value = self.ai_model_combo.currentData()
        if not value:
            return None
        return Path(str(value))

    def refresh_curve_lists(self, *, show_feedback: bool = False) -> None:
        previous_device = self.selected_device_curve.name if self.selected_device_curve is not None else "Default"
        previous_target = self.target_curve_combo.currentText() if hasattr(self, "target_curve_combo") else ""

        ensure_curve_dirs()
        self.device_curves = list_curves(DEVICE_CURVES_DIR, include_default=True, lazy=True)
        self.target_curves = list_curves(TARGET_CURVES_DIR, include_default=False, lazy=True)

        self.device_curve_combo.blockSignals(True)
        self.device_curve_combo.clear()
        device_index = 0
        for index, curve in enumerate(self.device_curves):
            self.device_curve_combo.addItem(self.device_curve_display_name(curve), index)
            if curve.name == previous_device:
                device_index = index
        self.device_curve_combo.setCurrentIndex(device_index)
        self.device_curve_combo.blockSignals(False)
        self.selected_device_curve = self._loaded_curve(self.device_curves[device_index]) if self.device_curves else None

        self.target_curve_combo.blockSignals(True)
        self.target_curve_combo.clear()
        self.device_curve_combo.setEnabled(bool(self.device_curves))
        self.rebuild_target_curve_combo(previous_target)
        self.update_graph()
        if show_feedback:
            self.show_toast(self.t("toast.curve_lists_updated"))

    def on_device_curve_changed(self, index: int) -> None:
        if 0 <= index < len(self.device_curves):
            previous_target = self.target_curve_combo.currentText()
            self.selected_device_curve = self._loaded_curve(self.device_curves[index])
            self.rebuild_target_curve_combo(previous_target)
            self.update_graph()

    def on_target_curve_changed(self, _index: int) -> None:
        self.update_graph()

    def rebuild_target_curve_combo(self, previous_target: str = "") -> None:
        selected_name = self.selected_device_curve.name if self.selected_device_curve is not None else ""
        options: list[FrequencyCurve] = []
        seen: set[str] = set()
        for curve in [*self.target_curves, *self.device_curves]:
            if curve.name == "Default" or (selected_name and curve.name == selected_name):
                continue
            key = curve.name.casefold()
            if key in seen:
                continue
            seen.add(key)
            options.append(curve)
        self.target_options = options

        self.target_curve_combo.blockSignals(True)
        self.target_curve_combo.clear()
        target_index = 0
        for index, curve in enumerate(self.target_options):
            self.target_curve_combo.addItem(curve.name, index)
            if curve.name == previous_target:
                target_index = index
        if self.target_options:
            self.target_curve_combo.setCurrentIndex(target_index)
        self.target_curve_combo.blockSignals(False)
        self.target_curve_combo.setEnabled(bool(self.target_options))
        self.run_autoeq_button.setEnabled(bool(self.target_options))

    def selected_device_response_db(self) -> np.ndarray:
        if self.selected_device_curve is None:
            return np.zeros_like(GRAPH_FREQS, dtype=np.float64)
        return self.selected_device_curve.response_db(GRAPH_FREQS)

    def selected_target_response_db(self) -> np.ndarray | None:
        if not hasattr(self, "target_curve_combo"):
            return None
        target_index = self.target_curve_combo.currentData()
        if target_index is None or not (0 <= int(target_index) < len(self.target_options)):
            return None
        return self._loaded_curve(self.target_options[int(target_index)]).response_db(GRAPH_FREQS)

    @staticmethod
    def _loaded_curve(curve: FrequencyCurve) -> FrequencyCurve:
        return curve.loaded()

    def refresh_presets(self) -> None:
        self.saved_presets = self.store.list_presets()
        self._updating = True
        self.current_selector.blockSignals(True)
        self.current_selector.clear()
        self.current_selector.addItem(self.new_preset_display_name(), NEW_PRESET_ID)
        selected_index = 0
        for preset in self.saved_presets:
            self.current_selector.addItem(preset.name, preset.id)
            if preset.id is not None and preset.id == self.current_preset.id:
                selected_index = self.current_selector.count() - 1
        self.current_selector.setCurrentIndex(selected_index)
        self.current_selector.blockSignals(False)
        self._updating = False
        self.rebuild_compare_menu()
        self.update_graph()

    def sync_current_selector_to_preset(self) -> None:
        if not hasattr(self, "current_selector"):
            return
        wanted_data = self.current_preset.id if self.current_preset.id is not None else NEW_PRESET_ID
        for index in range(self.current_selector.count()):
            if self.current_selector.itemData(index) == wanted_data:
                self._updating = True
                self.current_selector.blockSignals(True)
                self.current_selector.setCurrentIndex(index)
                self.current_selector.blockSignals(False)
                self._updating = False
                return

    def rebuild_compare_menu(self) -> None:
        self.compare_menu.clear()
        saved_ids = {preset.id for preset in self.saved_presets if preset.id is not None}
        if self.current_preset.id is not None:
            saved_ids.discard(self.current_preset.id)
        self.compare_ids = {preset_id for preset_id in self.compare_ids if preset_id in saved_ids}
        self.trim_compare_ids_to_limit()

    def ordered_compare_ids(self) -> list[int]:
        current_id = self.current_preset.id
        return [
            int(preset.id)
            for preset in self.saved_presets
            if preset.id is not None and preset.id != current_id and int(preset.id) in self.compare_ids
        ]

    def trim_compare_ids_to_limit(self) -> None:
        ordered_ids = self.ordered_compare_ids()
        allowed = set(ordered_ids[:MAX_COMPARE_PRESETS])
        self.compare_ids = {preset_id for preset_id in self.compare_ids if preset_id in allowed}

    def on_compare_toggled(self, checked: bool) -> None:
        action = self.sender()
        if not isinstance(action, QAction):
            return
        preset_id, _color = action.data()
        if checked:
            if len(self.ordered_compare_ids()) < MAX_COMPARE_PRESETS:
                self.compare_ids.add(int(preset_id))
        else:
            self.compare_ids.discard(int(preset_id))
        self.trim_compare_ids_to_limit()
        self.update_graph()

    def show_compare_menu(self) -> None:
        if hasattr(self, "_compare_popup") and self._compare_popup is not None:
            self._compare_popup.close()
        container = RoundedPopupPanel(self)
        container.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        left = self.compare_button.mapToGlobal(self.compare_button.rect().bottomLeft())
        right = self.export_button.mapToGlobal(self.export_button.rect().bottomRight())
        popup_width = max(180, right.x() - left.x())
        option_width = popup_width - 12
        option_content_width = max(1, option_width - POPUP_LIST_VIEWPORT_PADDING_X)
        container.setFixedWidth(popup_width)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        search = QLineEdit(container)
        search.setObjectName("comboSearch")
        search.setPlaceholderText(self.t("common.search"))
        list_widget = HoverListWidget(container)
        list_widget.setObjectName("compareSearchList")
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        search.setFixedWidth(option_width)
        list_widget.setFixedWidth(option_width)
        set_popup_list_rows(list_widget, 6)
        layout.addWidget(search)
        layout.addWidget(list_widget)

        def resize_compare_popup() -> None:
            margins = layout.contentsMargins()
            container.setFixedHeight(search.height() + layout.spacing() + list_widget.height() + margins.top() + margins.bottom())
            container.adjustSize()
            container.update_rounded_mask()

        current_id = self.current_preset.id
        presets = [
            preset
            for preset in self.saved_presets
            if preset.id is not None and preset.id != current_id
        ]
        all_preset_ids = {int(preset.id) for preset in presets if preset.id is not None}

        checkbox_by_row: dict[int, AieqCheckBox] = {}
        checkbox_by_preset_id: dict[int, AieqCheckBox] = {}
        item_by_preset_id: dict[int, QListWidgetItem] = {}
        all_checkbox: AieqCheckBox | None = None

        def set_hovered_compare_row(row: int) -> None:
            for checkbox_row, checkbox in checkbox_by_row.items():
                hovered = checkbox_row == row
                if bool(checkbox.property("searchMenuHovered")) == hovered:
                    continue
                checkbox.setProperty("searchMenuHovered", hovered)
                checkbox.update()

        def refresh_hovered_compare_row() -> None:
            set_hovered_compare_row(list_widget.row_at_global_pos(QCursor.pos()))

        list_widget.hovered_row_changed.connect(set_hovered_compare_row)
        list_widget.verticalScrollBar().valueChanged.connect(lambda _value: refresh_hovered_compare_row())

        def make_compare_checkbox(text: str, tooltip: str | None = None) -> AieqCheckBox:
            checkbox = AieqCheckBox(text)
            checkbox.setProperty("searchMenuCheckbox", True)
            checkbox.setProperty("searchMenuHovered", False)
            checkbox.setProperty("leftPadding", 8)
            checkbox.setProperty("textGap", 12)
            checkbox.setFixedWidth(option_content_width)
            checkbox.setFixedHeight(POPUP_OPTION_DEFAULT_ROW_HEIGHT)
            if tooltip:
                checkbox.setToolTip(tooltip_for_elided_text(tooltip, checkbox, display_text=text))
            else:
                checkbox.setToolTip("")
            return checkbox

        def add_checkbox_item(checkbox: AieqCheckBox, *, user_data: int | str) -> QListWidgetItem:
            row = list_widget.count()
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, user_data)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setSizeHint(QSize(option_content_width, POPUP_OPTION_DEFAULT_ROW_HEIGHT))
            list_widget.addItem(item)
            list_widget.setItemWidget(item, checkbox)
            list_widget.track_hover_widget(checkbox)
            checkbox_by_row[row] = checkbox
            return item

        def sync_compare_checkboxes() -> None:
            selected_count = len(self.compare_ids & all_preset_ids)
            selection_limit = min(MAX_COMPARE_PRESETS, len(all_preset_ids))
            limit_reached = selected_count >= MAX_COMPARE_PRESETS
            if all_checkbox is not None:
                all_checkbox.blockSignals(True)
                all_checkbox.setChecked(bool(all_preset_ids) and selected_count >= selection_limit)
                all_checkbox.blockSignals(False)
                all_checkbox.setEnabled(bool(all_preset_ids))
                all_checkbox.update()
            for preset_id, checkbox in checkbox_by_preset_id.items():
                selected = preset_id in self.compare_ids
                disabled = limit_reached and not selected
                checkbox.blockSignals(True)
                checkbox.setChecked(selected)
                checkbox.blockSignals(False)
                checkbox.setEnabled(not disabled)
                checkbox.update()
                item = item_by_preset_id.get(preset_id)
                if item is not None:
                    item.setFlags(
                        Qt.ItemFlag.NoItemFlags
                        if disabled
                        else Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )

        def toggle_all_compare(checked: bool) -> None:
            if checked:
                free_slots = max(0, MAX_COMPARE_PRESETS - len(self.compare_ids & all_preset_ids))
                for preset in presets:
                    if free_slots <= 0:
                        break
                    if preset.id is None:
                        continue
                    preset_id = int(preset.id)
                    if preset_id in self.compare_ids:
                        continue
                    self.compare_ids.add(preset_id)
                    free_slots -= 1
            else:
                self.compare_ids.clear()
            sync_compare_checkboxes()
            self.update_graph()

        def toggle_one_compare(preset_id: int, checked: bool) -> None:
            if checked:
                if len(self.compare_ids & all_preset_ids) >= MAX_COMPARE_PRESETS:
                    sync_compare_checkboxes()
                    return
                self.compare_ids.add(preset_id)
            else:
                self.compare_ids.discard(preset_id)
            sync_compare_checkboxes()
            self.update_graph()

        def click_compare_item(item: QListWidgetItem) -> None:
            value = item.data(Qt.ItemDataRole.UserRole)
            if value == "all":
                selection_limit = min(MAX_COMPARE_PRESETS, len(all_preset_ids))
                all_checked = bool(all_preset_ids) and len(self.compare_ids & all_preset_ids) >= selection_limit
                toggle_all_compare(not all_checked)
                return
            if value is None:
                return
            preset_id = int(value)
            toggle_one_compare(preset_id, preset_id not in self.compare_ids)

        list_widget.itemClicked.connect(click_compare_item)

        def populate(query: str = "") -> None:
            nonlocal checkbox_by_row, checkbox_by_preset_id, item_by_preset_id, all_checkbox
            list_widget.blockSignals(True)
            list_widget.clear()
            checkbox_by_row = {}
            checkbox_by_preset_id = {}
            item_by_preset_id = {}
            all_checkbox = None
            query = query.strip().casefold()
            if presets:
                all_checkbox = make_compare_checkbox(self.t("compare.all"))
                all_checkbox.setChecked(bool(all_preset_ids) and all_preset_ids.issubset(self.compare_ids))
                add_checkbox_item(all_checkbox, user_data="all")
                all_checkbox.clicked.connect(lambda _checked=False: click_compare_item(list_widget.item(0)))
            matches = [
                preset
                for preset in presets
                if not query or query in preset.name.casefold()
            ]
            if not matches:
                item = QListWidgetItem(self.t("compare.no_presets"))
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                list_widget.addItem(item)
                list_widget.blockSignals(False)
                refresh_hovered_compare_row()
                resize_compare_popup()
                return
            for preset in matches:
                preset_id = int(preset.id)
                display_name = elide_middle(preset.name)
                checkbox = make_compare_checkbox(display_name, preset.name)
                checkbox.setChecked(int(preset.id) in self.compare_ids)
                item = add_checkbox_item(checkbox, user_data=preset_id)
                item.setToolTip(tooltip_for_elided_text(preset.name, list_widget, display_text=display_name))
                checkbox.clicked.connect(lambda _checked=False, item=item: click_compare_item(item))
                checkbox_by_preset_id[preset_id] = checkbox
                item_by_preset_id[preset_id] = item
            sync_compare_checkboxes()
            list_widget.blockSignals(False)
            refresh_hovered_compare_row()
            resize_compare_popup()

        populate()
        search.textChanged.connect(populate)

        self._compare_popup = container
        container.destroyed.connect(lambda _obj=None: setattr(self, "_compare_popup", None))
        QTimer.singleShot(0, search.setFocus)
        resize_compare_popup()
        container.move(left)
        container.show()
        container.raise_()

    def on_compare_checkbox_toggled(self, preset_id: int, checked: bool) -> None:
        if checked:
            if len(self.ordered_compare_ids()) < MAX_COMPARE_PRESETS:
                self.compare_ids.add(int(preset_id))
        else:
            self.compare_ids.discard(int(preset_id))
        self.trim_compare_ids_to_limit()
        self.update_graph()

    def update_graph(self) -> None:
        self.trim_compare_ids_to_limit()
        device_db = self.selected_device_response_db()
        self.device_curve_item.setData(GRAPH_FREQS, device_db)
        target_db = self.selected_target_response_db()
        show_target = (
            target_db is not None
            and hasattr(self, "show_target_checkbox")
            and self.show_target_checkbox.isChecked()
        )
        if show_target and target_db is not None:
            self.target_curve_item.setData(GRAPH_FREQS, target_db)
            self.target_curve_item.setPen(pg.mkPen(TARGET_COLOR, width=2))
            self.target_curve_item.show()
            self.set_target_legend_visible(True)
        else:
            self.target_curve_item.hide()
            self.set_target_legend_visible(False)
        db = device_db + preset_response_db(self.current_preset, GRAPH_FREQS, DEFAULT_SAMPLE_RATE)
        self.current_curve.setPen(pg.mkPen(CURRENT_COLOR, width=3))
        self.current_curve.setData(GRAPH_FREQS, db, name=self.current_preset.name)

        for curve in self.compare_curves.values():
            self.plot.removeItem(curve)
        self.compare_curves.clear()

        displayed_compare_count = 0
        for idx, preset in enumerate(self.saved_presets):
            if preset.id is None or preset.id not in self.compare_ids or preset.id == self.current_preset.id:
                continue
            if displayed_compare_count >= MAX_COMPARE_PRESETS:
                break
            color = CURVE_COLORS[idx % len(CURVE_COLORS)]
            curve = self.plot.plot(
                GRAPH_FREQS,
                device_db + preset_response_db(preset, GRAPH_FREQS, DEFAULT_SAMPLE_RATE),
                pen=pg.mkPen(color, width=2),
                name=preset.name,
            )
            self.compare_curves[preset.id] = curve
            displayed_compare_count += 1

    def set_target_legend_visible(self, visible: bool) -> None:
        if not hasattr(self, "plot_legend"):
            return
        if visible == self.target_legend_visible:
            return
        if visible:
            self.plot_legend.addItem(self.target_curve_item, self.t("graph.target"))
        else:
            self.plot_legend.removeItem(self.target_curve_item)
        self.target_legend_visible = visible

    def populate_filter_editor(self) -> None:
        self._updating = True
        while self.filter_list_layout.count() > 1:
            item = self.filter_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.filter_rows = []
        for index, eq_filter in enumerate(self.current_preset.filters):
            self._add_filter_row(eq_filter, index)
        self.apply_filter_row_labels()
        if self.filter_rows:
            self.selected_filter_row = min(max(self.selected_filter_row, 0), len(self.filter_rows) - 1)
        else:
            self.selected_filter_row = -1
        self.filter_scroll.horizontalScrollBar().setValue(0)
        self.sync_filter_container_width()
        self.update_filter_selection()
        self.schedule_filter_container_sync()
        self._updating = False

    def schedule_filter_container_sync(self, *_args: object) -> None:
        if not hasattr(self, "filter_container"):
            return
        QTimer.singleShot(0, self.sync_filter_container_width)
        QTimer.singleShot(35, self.sync_filter_container_width)

    def sync_filter_container_width(self) -> None:
        if not hasattr(self, "filter_container"):
            return
        row_width = FilterEditorRow.FIXED_WIDTH
        spacing = self.filter_list_layout.spacing()
        margins = self.filter_list_layout.contentsMargins()
        row_count = len(self.filter_rows)
        content_width = margins.left() + margins.right()
        if row_count:
            content_width += row_count * row_width + max(0, row_count - 1) * spacing
        viewport_width = max(1, self.filter_scroll.viewport().width() - 2)
        width = max(viewport_width, content_width)
        height = max(self.filter_container.sizeHint().height(), self.filter_scroll.viewport().height())
        self.filter_container.setMinimumSize(0, 0)
        self.filter_container.setMaximumSize(16777215, 16777215)
        self.filter_container.resize(width, height)
        self.filter_container.updateGeometry()
        self.filter_container.update()
        self.filter_scroll.viewport().update()

    def _add_filter_row(self, eq_filter: EqFilter, index: int) -> None:
        row = FilterEditorRow(eq_filter, index)
        row.apply_labels(
            {
                "type": self.t("filter.type"),
                "gain": self.t("filter.gain"),
                "freq": self.t("filter.freq"),
                "q": self.t("filter.q"),
            },
            self.t("filter.enabled"),
            {filter_type: self.filter_type_label(filter_type) for filter_type in FILTER_TYPES},
        )
        row.installEventFilter(self)
        row.changed.connect(self.on_filters_changed)
        row.selected.connect(self.select_filter_row)
        row.type_icon.info_requested.connect(self.show_filter_type_info)
        row.param_info_requested.connect(lambda key, label, current_row=row: self.show_filter_param_info(current_row, key, label))
        self.filter_rows.append(row)
        self.filter_list_layout.insertWidget(
            self.filter_list_layout.count() - 1,
            row,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )

    def show_filter_type_info(self, filter_type: str, anchor: QPoint) -> None:
        if not self._can_open_filter_tooltip():
            return

        popup = RoundedPopupPanel(self)
        popup.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        popup.setFixedWidth(300)
        layout = QVBoxLayout(popup)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        image_label = QLabel(popup)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setObjectName("filterTypeInfoImage")
        pixmap = FilterTypeIcon.full_pixmap_for_type(filter_type)
        if not pixmap.isNull():
            image_label.setPixmap(
                pixmap.scaled(
                    220,
                    120,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        layout.addWidget(image_label)

        title_label = QLabel(self.filter_type_label(filter_type), popup)
        title_label.setObjectName("filterTypeInfoTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFixedHeight(22)
        layout.addWidget(title_label)

        description_label = TooltipTextLabel(
            self.filter_type_description(filter_type),
            300 - 24,
            popup,
        )
        layout.addWidget(description_label)

        popup.adjustSize()
        popup.update_rounded_mask()
        screen = self.screen() or QApplication.primaryScreen()
        sender = self.sender()
        trigger_height = sender.height() if isinstance(sender, QWidget) else 0
        position = QPoint(anchor.x(), anchor.y() - popup.height() - TOOLTIP_GAP)
        if screen is not None:
            available = screen.availableGeometry()
            position.setX(min(max(position.x(), available.left()), available.right() - popup.width() + 1))
            if position.y() < available.top():
                position.setY(anchor.y() + trigger_height + TOOLTIP_GAP)
        popup.move(position)
        self._filter_type_popup = popup
        popup.destroyed.connect(lambda _obj=None: self._on_filter_tooltip_destroyed("_filter_type_popup"))
        popup.show()
        popup.raise_()

    def show_filter_param_info(self, row: FilterEditorRow, param_key: str, label_widget: QWidget) -> None:
        if not self._can_open_filter_tooltip():
            return

        popup = RoundedPopupPanel(self)
        popup.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        left = row.type_label.mapToGlobal(QPoint(0, 0)).x()
        right = row.q_label.mapToGlobal(QPoint(row.q_label.width(), 0)).x()
        popup_width = max(240, right - left)
        label_top = label_widget.mapToGlobal(QPoint(0, 0)).y()
        graph_axis_y = self.plot.mapToGlobal(QPoint(0, max(0, self.plot.height() - 42))).y()
        top_y = graph_axis_y + TOOLTIP_GAP
        bottom_y = label_top - TOOLTIP_GAP
        fallback_position: QPoint | None = None
        if bottom_y <= top_y + 56:
            anchor = label_widget.mapToGlobal(QPoint(0, 0))
            fallback_position = QPoint(anchor.x(), anchor.y())

        popup.setFixedWidth(popup_width)
        layout = QVBoxLayout(popup)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title_label = QLabel(self.t(f"filter.{param_key}"), popup)
        title_label.setObjectName("filterTypeInfoTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFixedHeight(22)
        layout.addWidget(title_label)

        description_label = TooltipTextLabel(
            self.filter_param_description(param_key),
            popup_width - 24,
            popup,
        )
        layout.addWidget(description_label)

        popup.adjustSize()
        content_height = popup.sizeHint().height()
        if fallback_position is not None:
            popup_height = content_height
            top_y = fallback_position.y() - popup_height - TOOLTIP_GAP
        else:
            popup_height = content_height
            top_y = bottom_y - popup_height
        popup.setFixedHeight(popup_height)
        popup.update_rounded_mask()
        screen = self.screen() or QApplication.primaryScreen()
        position = QPoint(left, top_y)
        if screen is not None:
            available = screen.availableGeometry()
            position.setX(min(max(position.x(), available.left()), available.right() - popup.width() + 1))
            position.setY(min(max(position.y(), available.top()), available.bottom() - popup.height() + 1))
        popup.move(position)
        self._filter_param_popup = popup
        popup.destroyed.connect(lambda _obj=None: self._on_filter_tooltip_destroyed("_filter_param_popup"))
        popup.show()
        popup.raise_()

    def select_filter_row(self, row: FilterEditorRow) -> None:
        if row in self.filter_rows:
            self.selected_filter_row = self.filter_rows.index(row)
            self.update_filter_selection()

    def update_filter_selection(self) -> None:
        for index, row in enumerate(self.filter_rows):
            row.set_selected(index == self.selected_filter_row)

    def read_filters_from_editor(self) -> list[EqFilter]:
        return [row.to_filter() for row in self.filter_rows]

    def on_filters_changed(self) -> None:
        if self._updating:
            return
        self.current_preset.filters = self.read_filters_from_editor()
        current_signature = self.preset_filter_signature(self.current_preset)
        if current_signature != self._clean_preset_filter_signature and self.current_preset.id is not None:
            self.current_preset = self.current_preset.clone(name=f"{self.current_preset.name} (ред.)", keep_id=False)
            self.refresh_presets()
        elif current_signature == self._clean_preset_filter_signature:
            self.restore_clean_preset_identity(refresh=True)
        self.update_graph()
        if self.audio_engine.is_running:
            self.schedule_audio_update()

    def add_filter(self) -> None:
        self.current_preset.filters.append(EqFilter())
        self.selected_filter_row = len(self.current_preset.filters) - 1
        self.populate_filter_editor()
        self.on_filters_changed()

    def remove_selected_filter(self) -> None:
        row = self.selected_filter_row
        if row < 0 or row >= len(self.current_preset.filters):
            return
        del self.current_preset.filters[row]
        self.selected_filter_row = min(row, len(self.current_preset.filters) - 1)
        self.populate_filter_editor()
        self.on_filters_changed()

    def clear_filters(self) -> None:
        self.current_preset.filters = []
        self.selected_filter_row = -1
        self.populate_filter_editor()
        self.on_filters_changed()

    def schedule_audio_update(self) -> None:
        self.audio_update_timer.start(90)

    def apply_audio_preset(self) -> None:
        if self.audio_engine.is_running:
            self.audio_engine.update_preset(self.current_preset)

    @staticmethod
    def editor_visible_filter_dict(eq_filter: EqFilter) -> dict[str, object]:
        clean = eq_filter.sanitized()
        return {
            "type": clean.type,
            "freq": round(float(clean.freq)),
            "q": round(float(clean.q), 3),
            "gain": round(float(clean.gain), 2),
            "enabled": bool(clean.enabled),
        }

    @staticmethod
    def filter_signature(filters: list[EqFilter]) -> str:
        clean_filters = [MainWindow.editor_visible_filter_dict(eq_filter) for eq_filter in filters]
        return json.dumps(clean_filters, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def preset_filter_signature(preset: Preset) -> str:
        return MainWindow.filter_signature(preset.filters)

    def mark_current_preset_clean(self) -> None:
        self._clean_preset_filter_signature = self.preset_filter_signature(self.current_preset)
        self._clean_preset_id = self.current_preset.id
        self._clean_preset_name = self.current_preset.name

    def restore_clean_preset_identity(self, *, refresh: bool) -> None:
        if self.current_preset.id is not None or self._clean_preset_id is None:
            return
        self.current_preset.id = self._clean_preset_id
        self.current_preset.name = self._clean_preset_name
        if refresh:
            self.refresh_presets()
            self.sync_current_selector_to_preset()

    def current_filter_signature(self) -> str:
        if hasattr(self, "filter_rows") and not self._updating:
            self.current_preset.filters = self.read_filters_from_editor()
        return self.preset_filter_signature(self.current_preset)

    def has_unsaved_changes(self) -> bool:
        current_signature = self.current_filter_signature()
        if current_signature == self._clean_preset_filter_signature:
            self.restore_clean_preset_identity(refresh=True)
            return False
        return True

    def confirm_or_save_before_switch(self) -> bool:
        if not self.has_unsaved_changes():
            return True
        answer = ask_custom_question(
            self,
            self.t("dialog.save_changes_title"),
            self.t("dialog.save_changes_text"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            return self.save_current_preset()
        return True

    def new_flat_preset(self) -> None:
        self.current_preset = flat_preset()
        self.mark_current_preset_clean()
        self.populate_filter_editor()
        self.refresh_presets()
        self.apply_audio_preset()

    def load_current_from_selector(self, index: int) -> None:
        if self._updating:
            return
        preset_id = self.current_selector.itemData(index)
        if preset_id == NEW_PRESET_ID and self.current_preset.id is None and not self.has_unsaved_changes():
            return
        if preset_id != NEW_PRESET_ID and self.current_preset.id == int(preset_id):
            return

        if not self.confirm_or_save_before_switch():
            self.refresh_presets()
            return

        if preset_id == NEW_PRESET_ID:
            self.current_preset = flat_preset()
        else:
            preset = self.store.get_preset(int(preset_id))
            if preset is None:
                self.refresh_presets()
                return
            self.current_preset = preset.clone(keep_id=True)
        self.mark_current_preset_clean()
        self.populate_filter_editor()
        if isinstance(preset_id, int):
            self.compare_ids.discard(int(preset_id))
        self.rebuild_compare_menu()
        self.refresh_presets()
        self.update_graph()
        self.apply_audio_preset()

    def delete_current_preset(self) -> None:
        if self.current_preset.id is None:
            self.show_toast(self.t("toast.current_preset_unsaved"))
            return
        preset_id = self.current_preset.id
        answer = ask_custom_question(
            self,
            self.t("dialog.delete_preset_title"),
            self.t("dialog.delete_preset_text").format(name=self.current_preset.name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.store.delete(preset_id)
        self.compare_ids.discard(preset_id)
        self.current_preset = flat_preset()
        self.mark_current_preset_clean()
        self.populate_filter_editor()
        self.refresh_presets()
        self.apply_audio_preset()
        self.show_toast(self.t("toast.preset_deleted"))

    def save_current_preset(self) -> bool:
        if not self.current_preset.filters:
            return False
        name, accepted = get_custom_text(self, self.t("dialog.save_preset_title"), self.t("dialog.preset_name"), text=self.current_preset.name)
        if not accepted:
            return False
        name = name.strip() or self.current_preset.name
        if self.is_reserved_preset_name(name):
            show_custom_message(self, self.t("dialog.save_preset_title"), self.t("dialog.new_reserved"))
            return False
        existing = self.store.get_preset_by_name(name)
        if existing is not None:
            answer = ask_custom_question(
                self,
                self.t("dialog.overwrite_preset_title"),
                self.t("dialog.overwrite_preset_text").format(name=name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
            overwrite = self.current_preset.clone(name=name, keep_id=False)
            overwrite.id = existing.id
            saved = self.store.update(overwrite)
        else:
            saved = self.store.save_new(self.current_preset, name=name)
        self.current_preset = saved.clone(keep_id=True)
        self.mark_current_preset_clean()
        self.refresh_presets()
        self.show_toast(self.t("toast.preset_saved"))
        return True

    def is_reserved_preset_name(self, name: str) -> bool:
        return name.strip().casefold() == "new"

    def next_available_preset_name(self, name: str) -> str:
        base = name.strip() or "Preset"
        existing = {preset.name.casefold() for preset in self.store.list_presets()}
        if base.casefold() not in existing:
            return base
        index = 2
        while f"{base} {index}".casefold() in existing:
            index += 1
        return f"{base} {index}"

    def timestamp_parts(self) -> tuple[str, str]:
        now = datetime.now()
        return now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S")

    def ai_preset_name(self) -> str:
        date, time = self.timestamp_parts()
        return f"AIEQ {date} | {time}"

    def autoeq_preset_name(self, mode: str, origin: str, target: str) -> str:
        date, time = self.timestamp_parts()
        return f"AutoEQ {mode} | {date} | {time} – {origin} to {target}"

    def save_generated_preset(self, preset: Preset, *, name: str | None = None) -> Preset:
        preset_name = name or preset.name
        return self.store.save_new(preset, name=self.next_available_preset_name(preset_name))

    def import_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.t("dialog.import_preset_title"), str(Path.cwd()), "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.current_preset = Preset.from_dict(data)
            empty_signature = self.preset_filter_signature(flat_preset())
            self._clean_preset_filter_signature = empty_signature
            self._clean_preset_id = None
            self._clean_preset_name = "New"
            self.populate_filter_editor()
            self.refresh_presets()
            self.apply_audio_preset()
        except Exception as exc:  # noqa: BLE001
            show_custom_message(self, self.t("button.import"), f"{self.t('dialog.import_failed')}\n{exc}")

    def export_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, self.t("dialog.export_preset_title"), f"{self.current_preset.name}.json", "JSON (*.json)")
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        try:
            target.write_text(json.dumps(self.current_preset.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            show_custom_message(self, self.t("button.export"), f"{self.t('dialog.export_failed')}\n{exc}")

    def run_autoeq(self) -> None:
        if self._autoeq_thread is not None and self._autoeq_thread.isRunning():
            self.show_toast(self.t("toast.autoeq_still_running"))
            return
        if self.selected_device_curve is None:
            self.show_toast(self.t("toast.choose_device"))
            return
        target_index = self.target_curve_combo.currentData()
        if target_index is None or not (0 <= int(target_index) < len(self.target_options)):
            self.show_toast(self.t("toast.choose_target"))
            return
        target_curve = self._loaded_curve(self.target_options[int(target_index)])
        device_curve = self._loaded_curve(self.selected_device_curve)
        backend = str(self.autoeq_backend_combo.currentData() or "local")
        mode_label = self.autoeq_backend_combo.currentText()
        self.run_autoeq_button.setEnabled(False)
        self.show_toast(self.t("toast.autoeq_thinking"))
        self._autoeq_thread = QThread(self)
        self._autoeq_worker = AutoEqWorker(device_curve, target_curve, backend, mode_label)
        self._autoeq_worker.moveToThread(self._autoeq_thread)
        self._autoeq_thread.started.connect(self._autoeq_worker.run)
        self._autoeq_worker.finished.connect(self.on_autoeq_finished)
        self._autoeq_worker.failed.connect(self.on_autoeq_failed)
        self._autoeq_worker.finished.connect(self._autoeq_thread.quit)
        self._autoeq_worker.failed.connect(self._autoeq_thread.quit)
        self._autoeq_thread.finished.connect(self._autoeq_worker.deleteLater)
        self._autoeq_thread.finished.connect(self._autoeq_thread.deleteLater)
        self._autoeq_thread.finished.connect(self.on_autoeq_thread_finished)
        self._autoeq_thread.start()

    def on_autoeq_finished(self, result: AutoEqPresetResult, mode_label: str, origin_name: str, target_name: str) -> None:
        preset = result.preset
        preset_name = self.autoeq_preset_name(
            mode_label,
            origin_name,
            target_name,
        )
        saved = self.save_generated_preset(preset, name=preset_name)
        self.current_preset = saved.clone(keep_id=True)
        self.mark_current_preset_clean()
        self.show_target_checkbox.setChecked(True)
        self.populate_filter_editor()
        self.refresh_presets()
        self.apply_audio_preset()
        self.show_toast(self.t("toast.autoeq_applied"))

    def on_autoeq_failed(self, message: str) -> None:
        self.run_autoeq_button.setToolTip(message)
        self.show_toast(self.t("toast.autoeq_unavailable"), timeout_ms=4200)

    def on_autoeq_thread_finished(self) -> None:
        self._autoeq_thread = None
        self._autoeq_worker = None
        if hasattr(self, "run_autoeq_button"):
            self.run_autoeq_button.setEnabled(bool(self.target_options))

    def set_audio_running_ui(self, running: bool) -> None:
        self.audio_icon_label.setProperty("running", running)
        self.audio_icon_label.style().unpolish(self.audio_icon_label)
        self.audio_icon_label.style().polish(self.audio_icon_label)
        self.audio_icon_label.update()
        self.audio_button.setToolTip(self.t("audio.stop") if running else self.t("audio.start"))
        self.audio_button.setProperty("running", running)
        self.audio_button.style().unpolish(self.audio_button)
        self.audio_button.style().polish(self.audio_button)
        self.audio_button.update()
        self.refresh_devices_button.setEnabled(not running)
        for label in (self.input_label, self.output_label, self.sample_rate_label, self.audio_dtype_label):
            label.setEnabled(not running)
        self.input_combo.setEnabled(not running)
        self.output_combo.setEnabled(not running)
        self.sample_rate_combo.setEnabled(not running and self.sample_rate_combo.count() > 0)
        self.audio_dtype_combo.setEnabled(not running and self.audio_dtype_combo.count() > 0)
        self.refresh_latency_settings_state()

    def update_audio_latency_label(self) -> None:
        latency_ms = self.audio_engine.output_latency_ms
        if latency_ms is None:
            self.status_label.setText("--")
            self.status_label.adjustSize()
            return
        self.status_label.setText(f"{latency_ms:.1f} ms")
        self.status_label.adjustSize()

    def toggle_audio(self) -> None:
        if self.audio_engine.is_running:
            self.audio_engine.stop()
            self.audio_latency_timer.stop()
            self.set_audio_running_ui(False)
            self.update_audio_latency_label()
            return

        self.refresh_devices(show_feedback=False)
        input_device = self._selected_device(self.input_combo, self.input_devices)
        output_device = self._selected_device(self.output_combo, self.output_devices)
        if input_device is None or output_device is None:
            show_custom_message(self, self.t("settings.audio"), self.t("dialog.choose_io"))
            return
        try:
            latency, custom_latency = self.selected_audio_latency()
            self.audio_engine.set_latency(latency, custom=custom_latency)
            self.audio_engine.start(
                input_device,
                output_device,
                self.current_preset,
                sample_rate=self.selected_sample_rate(),
                dtype=self.selected_audio_dtype(),
            )
            self.set_audio_running_ui(True)
            self.update_audio_latency_label()
            self.audio_latency_timer.start(1000)
        except Exception as exc:  # noqa: BLE001
            self.audio_engine.stop()
            self.audio_latency_timer.stop()
            self.set_audio_running_ui(False)
            self.update_audio_latency_label()
            show_custom_message(self, self.t("settings.audio"), f"{self.t('dialog.audio_start_failed')}\n{exc}")

    def _selected_device(self, combo: QComboBox, devices: list[AudioDevice]) -> AudioDevice | None:
        index = combo.currentData()
        for device in devices:
            if device.index == index:
                return device
        return None

    def append_chat(self, author: str, text: str) -> None:
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        color = USER_CHAT_COLOR if author == self.t("chat.you") else CURRENT_COLOR
        self.chat_history.append(f'<p><b style="color:{color}">{author}</b><br>{safe}</p>')

    def show_chat_menu(self) -> None:
        self.refresh_chat_sessions()
        if self._chat_popup is not None:
            self._chat_popup.close()
        container = RoundedPopupPanel(self)
        container.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        popup_width = max(220, self.chat_input.width())
        container.setFixedWidth(popup_width)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        list_widget = HoverListWidget(container)
        list_widget.setObjectName("comboSearchList")
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setUniformItemSizes(True)
        list_widget.setItemDelegate(PopupOptionDelegate(list_widget))
        list_widget.setFixedWidth(popup_width - 12)
        list_widget.setMaximumHeight(220)

        search = QLineEdit(container)
        search.setObjectName("comboSearch")
        search.setPlaceholderText(self.t("common.search"))
        search.setFixedWidth(popup_width - 12)

        layout.addWidget(list_widget)
        layout.addWidget(search)

        def populate(query: str = "") -> None:
            list_widget.clear()
            query = query.strip().casefold()
            matches = [
                session
                for session in self.chat_sessions
                if not query or query in session.title.casefold()
            ]
            if not matches:
                item = QListWidgetItem(self.t("chat.no_saved"))
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                list_widget.addItem(item)
                return
            for session in matches:
                item = QListWidgetItem(session.title)
                item.setData(Qt.ItemDataRole.UserRole, session.id)
                item.setToolTip(tooltip_for_elided_text(session.title, list_widget))
                list_widget.addItem(item)

        def choose(item: QListWidgetItem) -> None:
            chat_id = item.data(Qt.ItemDataRole.UserRole)
            if chat_id is None:
                return
            container.close()
            self.load_chat_session(int(chat_id))

        populate()
        search.textChanged.connect(populate)
        list_widget.itemClicked.connect(choose)

        set_popup_list_rows(list_widget, 3)
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setToolTip(tooltip_for_elided_text(item.text(), list_widget))
        container.adjustSize()
        container.update_rounded_mask()

        top = self.chat_menu_button.mapToGlobal(QPoint(0, 0))
        popup_gap = 0
        left = self.chat_input.mapToGlobal(QPoint(0, 0))
        position = QPoint(left.x(), top.y() - container.height() - popup_gap)
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None and position.y() < screen.availableGeometry().top():
            position.setY(top.y() + self.chat_menu_button.height() + popup_gap)
        container.move(position)
        self._chat_popup = container
        container.destroyed.connect(lambda _obj=None: setattr(self, "_chat_popup", None))
        container.show()
        container.raise_()
        QTimer.singleShot(0, search.setFocus)

    def refresh_chat_sessions(self) -> None:
        self.chat_sessions = self.chat_store.list_sessions()
        if not hasattr(self, "chat_menu"):
            return
        self.chat_menu.clear()
        if not self.chat_sessions:
            action = QAction(self.t("chat.no_saved"), self.chat_menu)
            action.setEnabled(False)
            self.chat_menu.addAction(action)
            return
        for session in self.chat_sessions:
            action = QAction(session.title, self.chat_menu)
            action.setCheckable(True)
            action.setChecked(session.id == self.current_chat_id)
            action.triggered.connect(lambda _checked=False, chat_id=session.id: self.load_chat_session(chat_id))
            self.chat_menu.addAction(action)

    def restore_chat_session(self) -> None:
        last_id = self.settings.value("ai/current_chat_id", None)
        try:
            chat_id = int(last_id) if last_id is not None else None
        except (TypeError, ValueError):
            chat_id = None
        if chat_id is not None and self.chat_store.get_session(chat_id) is not None:
            self.load_chat_session(chat_id, show_feedback=False)
            return
        if self.chat_sessions:
            self.load_chat_session(self.chat_sessions[0].id, show_feedback=False)
        else:
            self.render_chat_messages()

    def load_chat_session(self, chat_id: int | None, *, show_feedback: bool = True) -> None:
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast(self.t("toast.ai_still_answering"))
            return
        if chat_id is None:
            self.start_new_chat(show_feedback=show_feedback)
            return
        session = self.chat_store.get_session(chat_id)
        if session is None:
            self.show_toast(self.t("toast.chat_not_found"))
            self.refresh_chat_sessions()
            return
        self.ai_service.clear_context()
        self.current_chat_id = session.id
        self.current_chat_context_full = session.context_full
        self.chat_messages = list(session.messages)
        self.settings.setValue("ai/current_chat_id", self.current_chat_id)
        self.render_chat_messages()
        self.refresh_chat_sessions()
        if show_feedback:
            self.show_toast(self.t("toast.chat_opened"))

    def start_new_chat(self, *, show_feedback: bool = True) -> None:
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast(self.t("toast.ai_still_answering"))
            return
        self.ai_service.clear_context()
        self.current_chat_id = None
        self.current_chat_context_full = False
        self.chat_messages = []
        self.settings.remove("ai/current_chat_id")
        self.render_chat_messages()
        self.refresh_chat_sessions()
        if show_feedback:
            self.show_toast(self.t("toast.new_chat"))

    def delete_current_chat(self) -> None:
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast(self.t("toast.ai_still_answering"))
            return
        if self.current_chat_id is None:
            self.start_new_chat()
            return
        answer = ask_custom_question(
            self,
            self.t("dialog.delete_chat_title"),
            self.t("dialog.delete_chat_text"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.chat_store.delete(self.current_chat_id)
        self.show_toast(self.t("toast.chat_deleted"))
        self.start_new_chat(show_feedback=False)

    def render_chat_messages(self) -> None:
        self.chat_history.clear()
        self.append_chat("AIEQ", self.t("chat.intro"))
        for message in self.chat_messages:
            author = self.t("chat.you") if message.get("role") == "user" else "AIEQ"
            self.append_chat(author, str(message.get("content", "")))
        self.update_chat_context_state()

    def ensure_current_chat(self, first_user_text: str) -> None:
        if self.current_chat_id is not None:
            return
        session = self.chat_store.save_new(chat_title_from_first_user_message(first_user_text), [])
        self.current_chat_id = session.id
        self.current_chat_context_full = False
        self.settings.setValue("ai/current_chat_id", self.current_chat_id)
        self.refresh_chat_sessions()

    def save_current_chat(self) -> None:
        if self.current_chat_id is None:
            return
        session = ChatSession(
            id=self.current_chat_id,
            title=self.current_chat_title(),
            messages=list(self.chat_messages),
            context_full=self.current_chat_context_full,
        )
        self.chat_store.update(session)
        self.refresh_chat_sessions()

    def current_chat_title(self) -> str:
        if self.current_chat_id is not None:
            for session in self.chat_sessions:
                if session.id == self.current_chat_id:
                    return session.title
        for message in self.chat_messages:
            if message.get("role") == "user":
                return chat_title_from_first_user_message(str(message.get("content", "")))
        return chat_title_from_first_user_message("")

    def mark_current_chat_context_full(self) -> None:
        self.current_chat_context_full = True
        self.save_current_chat()
        self.update_chat_context_state()
        self.show_toast(self.t("toast.context_full"), timeout_ms=4200)

    def update_chat_context_state(self) -> None:
        running = self._ai_thread is not None and self._ai_thread.isRunning()
        blocked = self.current_chat_context_full
        self.chat_input.setEnabled(not blocked)
        self.send_button.setEnabled(not blocked and not running)
        if blocked:
            self.chat_input.setPlaceholderText(self.t("chat.context_full_placeholder"))
        else:
            self.chat_input.setPlaceholderText(self.t("chat.placeholder"))

    def show_toast(self, text: str, timeout_ms: int = 2200) -> None:
        self.toast_label.setText(text)
        self.toast_label.setMinimumWidth(0)
        self.toast_label.setMaximumWidth(16777215)
        self.toast_label.adjustSize()
        max_width = max(120, self.width() - 80)
        width = min(self.toast_label.sizeHint().width(), max_width)
        self.toast_label.setFixedWidth(width)
        self.toast_label.adjustSize()
        self._position_toast()
        self.toast_label.show()
        self.toast_label.raise_()
        self.toast_timer.start(timeout_ms)

    def hide_toast(self) -> None:
        self.toast_label.hide()

    def _position_toast(self) -> None:
        if not hasattr(self, "toast_label"):
            return
        central = self.centralWidget()
        width = central.width() if central is not None else self.width()
        x = max(0, (width - self.toast_label.width()) // 2)
        if hasattr(self, "main_splitter") and central is not None:
            top = self.main_splitter.mapTo(central, QPoint(0, 0)).y()
            y = max(TITLE_BAR_HEIGHT + 6, top + 2)
        else:
            y = TITLE_BAR_HEIGHT + 8
        self.toast_label.move(x, y)

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if hasattr(self, "title_drag_widgets") and obj in self.title_drag_widgets:
            if self.handle_title_bar_event(event):
                return True
        if hasattr(self, "filter_scroll") and self.is_filter_scroll_object(obj):
            if event.type() in {QEvent.Type.Resize, QEvent.Type.Show}:
                self.schedule_filter_container_sync()
            elif event.type() == QEvent.Type.Wheel and self.scroll_filters_with_wheel(event):
                return True
        return super().eventFilter(obj, event)

    def is_filter_scroll_object(self, obj) -> bool:
        if obj is self.filter_scroll.viewport():
            return True
        if hasattr(self, "filter_container") and obj is self.filter_container:
            return True
        return hasattr(self, "filter_rows") and obj in self.filter_rows

    def scroll_filters_with_wheel(self, event) -> bool:
        bar = self.filter_scroll.horizontalScrollBar()
        if bar.maximum() <= 0:
            return False
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        if not pixel_delta.isNull():
            delta = pixel_delta.x() or pixel_delta.y()
            step = -delta
        else:
            delta = angle_delta.x() or angle_delta.y()
            if delta == 0:
                return False
            step = int(-delta / 120 * bar.singleStep())
        if step == 0:
            step = bar.singleStep() if delta < 0 else -bar.singleStep()
        bar.setValue(bar.value() + step)
        event.accept()
        return True

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.position_resize_handles()
        self.update_window_mask()
        if not self.is_window_maximized() and not self.isMinimized() and not self._is_screen_wide_geometry(self.geometry()):
            self._normal_geometry = self.geometry()
        self.position_title_logo()
        self.sync_filter_container_width()
        self.schedule_filter_container_sync()
        self._position_toast()

    def send_chat(self) -> None:
        text = self.chat_input.toPlainText().strip()
        if not text:
            return
        if self.current_chat_context_full:
            self.show_toast(self.t("toast.context_full"), timeout_ms=4200)
            self.update_chat_context_state()
            return
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast(self.t("toast.ai_still_answering"))
            return
        self.apply_ai_runtime_settings()
        history_for_model = list(self.chat_messages)
        self.ensure_current_chat(text)
        self.chat_input.clear()
        self.send_button.setEnabled(False)
        self.append_chat(self.t("chat.you"), text)
        self.chat_messages.append({"role": "user", "content": text})
        self.save_current_chat()
        self.show_toast(self.t("toast.ai_thinking"))

        self._ai_thread = QThread(self)
        self._ai_worker = AiWorker(
            self.ai_service,
            text,
            self.current_preset.clone(keep_id=True),
            saved_presets=[preset.clone(keep_id=True) for preset in self.saved_presets],
            model_path=self.selected_ai_model_path(),
            device_curve=self.selected_device_curve,
            chat_history=history_for_model,
        )
        self._ai_worker.moveToThread(self._ai_thread)
        self._ai_thread.started.connect(self._ai_worker.run)
        self._ai_worker.finished.connect(self.on_ai_finished)
        self._ai_worker.finished.connect(self._ai_thread.quit)
        self._ai_worker.finished.connect(self._ai_worker.deleteLater)
        self._ai_thread.finished.connect(self.on_ai_thread_finished)
        self._ai_thread.finished.connect(self._ai_thread.deleteLater)
        self._ai_thread.start()

    def on_ai_finished(self, result: AiPresetResult) -> None:
        if result.preset is None:
            self.append_chat("AIEQ", result.assistant_message)
            self.chat_messages.append({"role": "assistant", "content": result.assistant_message})
            if result.raw_json and self.ai_service.is_context_limit_message(result.raw_json):
                self.mark_current_chat_context_full()
            else:
                self.save_current_chat()
                self.send_button.setEnabled(not self.current_chat_context_full)
            return
        saved = self.save_generated_preset(result.preset, name=self.ai_preset_name())
        self.current_preset = saved.clone(keep_id=True)
        self.mark_current_preset_clean()
        self.populate_filter_editor()
        self.refresh_presets()
        self.append_chat("AIEQ", result.assistant_message)
        self.chat_messages.append({"role": "assistant", "content": result.assistant_message})
        self.show_toast(self.t("toast.preset_applied"))
        self.save_current_chat()
        self.send_button.setEnabled(not self.current_chat_context_full)
        self.apply_audio_preset()

    def on_ai_thread_finished(self) -> None:
        self._ai_thread = None
        self._ai_worker = None
        self.update_chat_context_state()


def apply_aieq_style() -> None:
    MainWindow._apply_style()

