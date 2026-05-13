from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDial,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
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
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QWidgetAction,
    QVBoxLayout,
    QWidget,
)

from .ai import AiEqualizerService, AiPresetResult, list_local_models
from .autoeq_service import AutoEqOfficialUnavailable, build_autoeq_preset_result
from .audio import AudioDevice, AudioEngine, AudioStreamSetting, list_audio_devices, list_supported_stream_settings, refresh_audio_backend
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
CHAT_INTRO_TEXT = "Опиши, что хочется изменить в звуке. Я сохраню ответ как новый пресет и применю его."
NEW_PRESET_ID = "__new__"
DEFAULT_WINDOW_WIDTH = 1480
DEFAULT_WINDOW_HEIGHT = 900
DEFAULT_SPLITTER_SIZES = (1040, 440)
LEGEND_LABEL_MAX_CHARS = 34


def elide_middle(text: str, max_chars: int = LEGEND_LABEL_MAX_CHARS) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    if max_chars <= 3:
        return "." * max_chars
    left = (max_chars - 3 + 1) // 2
    right = max_chars - 3 - left
    return f"{clean[:left]}...{clean[-right:]}"


class FrequencyAxisItem(pg.AxisItem):
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


class HoverLegendItem(pg.LegendItem):
    COLLAPSED_WIDTH = 24
    COLLAPSED_HEIGHT = 24
    MAX_ROWS_PER_COLUMN = 6
    EXPANDED_PADDING_X = 18
    EXPANDED_PADDING_Y = 14

    def __init__(self, *args, **kwargs) -> None:
        self._collapsed = False
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.set_collapsed(True)

    def addItem(self, item, name):  # type: ignore[override]
        display_name = elide_middle(str(name))
        super().addItem(item, display_name)
        if self.items:
            _sample, label = self.items[-1]
            label.setToolTip(str(name))
        self._reflow_items()
        self.set_collapsed(self._collapsed)

    def removeItem(self, item):  # type: ignore[override]
        super().removeItem(item)
        self._reflow_items()
        self.set_collapsed(self._collapsed)

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
        ev.acceptDrags(Qt.MouseButton.LeftButton)

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

    def paint(self, p, *args):  # type: ignore[override]
        super().paint(p, *args)
        if self._collapsed:
            p.setPen(pg.mkPen("#59616e"))
            p.setBrush(pg.mkBrush(17, 19, 24, 230))
            p.drawRect(self.boundingRect())
            font = p.font()
            font.setPointSize(13)
            p.setFont(font)
            p.setPen(pg.mkPen("#cfd6df"))
            p.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, "~")


class SearchableComboBox(QComboBox):
    def __init__(self, *, empty_text: str) -> None:
        super().__init__()
        self.empty_text = empty_text
        self._search_popup: QMenu | None = None

    def showPopup(self) -> None:  # type: ignore[override]
        menu = QMenu(self)
        menu.setObjectName("searchPopup")
        container = QWidget(menu)
        popup_width = max(80, self.width())
        container.setFixedWidth(popup_width)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        search = QLineEdit(container)
        search.setObjectName("comboSearch")
        search.setPlaceholderText("Поиск")
        list_widget = QListWidget(container)
        list_widget.setObjectName("comboSearchList")
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        search.setFixedWidth(popup_width - 12)
        list_widget.setFixedWidth(popup_width - 12)
        list_widget.setMaximumHeight(260)
        layout.addWidget(search)
        layout.addWidget(list_widget)

        def populate(query: str = "") -> None:
            list_widget.clear()
            query = query.strip().casefold()
            matches = [
                index
                for index in range(self.count())
                if not query or query in self.itemText(index).casefold()
            ]
            if not matches:
                item = QListWidgetItem(self.empty_text)
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                list_widget.addItem(item)
                return
            for index in matches:
                item = QListWidgetItem(self.itemText(index))
                item.setData(Qt.ItemDataRole.UserRole, index)
                if index == self.currentIndex():
                    item.setSelected(True)
                list_widget.addItem(item)

        def choose(item: QListWidgetItem) -> None:
            index = item.data(Qt.ItemDataRole.UserRole)
            if index is None:
                return
            self.setCurrentIndex(int(index))
            menu.close()

        populate()
        search.textChanged.connect(populate)
        list_widget.itemClicked.connect(choose)

        action = QWidgetAction(menu)
        action.setDefaultWidget(container)
        menu.addAction(action)
        self._search_popup = menu
        QTimer.singleShot(0, search.setFocus)
        menu.exec(self.mapToGlobal(self.rect().bottomLeft()))
        self._search_popup = None


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


class FilterEditorRow(QFrame):
    changed = Signal()
    selected = Signal(object)
    FIXED_WIDTH = 458
    FIXED_HEIGHT = 122

    def __init__(self, eq_filter: EqFilter, index: int) -> None:
        super().__init__()
        self.index = index
        self._syncing = False
        self.setObjectName("filterRow")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(self.FIXED_WIDTH, self.FIXED_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(5)

        self.enabled_check = QCheckBox()
        self.enabled_check.setToolTip("Вкл")
        self.enabled_check.setFixedWidth(24)
        self.enabled_check.setChecked(eq_filter.enabled)
        self.type_combo = SearchableComboBox(empty_text="Нет типов")
        self.type_combo.addItems(FILTER_TYPES)
        self.type_combo.setCurrentText(eq_filter.type)
        self.type_combo.setFixedWidth(76)

        self.gain_dial = QDial()
        self.gain_dial.setRange(-2400, 2400)
        self.gain_dial.setSingleStep(25)
        self.gain_dial.setPageStep(100)
        self.gain_dial.setNotchesVisible(True)
        self.gain_dial.setFixedSize(58, 58)
        self.gain_dial.setValue(int(round(eq_filter.gain * 100)))

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(-24.0, 24.0)
        self.gain_spin.setDecimals(2)
        self.gain_spin.setSingleStep(0.25)
        self.gain_spin.setKeyboardTracking(False)
        self.gain_spin.setValue(eq_filter.gain)
        self.gain_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gain_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.gain_spin.setFixedWidth(82)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20.0, 20000.0)
        self.freq_spin.setDecimals(0)
        self.freq_spin.setSingleStep(10.0)
        self.freq_spin.setKeyboardTracking(False)
        self.freq_spin.setValue(eq_filter.freq)
        self.freq_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.freq_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.freq_spin.setFixedWidth(92)

        self.q_spin = QDoubleSpinBox()
        self.q_spin.setRange(0.1, 18.0)
        self.q_spin.setDecimals(3)
        self.q_spin.setSingleStep(0.01)
        self.q_spin.setKeyboardTracking(False)
        self.q_spin.setValue(eq_filter.q)
        self.q_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.q_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.q_spin.setFixedWidth(76)

        index_label = QLabel(f"{index + 1:02d}")
        index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(index_label, 0, 0, 2, 1)
        layout.addWidget(self.enabled_check, 0, 1)
        layout.addWidget(self._param_label("type"), 0, 2)
        layout.addWidget(self.type_combo, 1, 2)
        layout.addWidget(self._param_label("gain"), 0, 3, 1, 2)
        layout.addWidget(self.gain_dial, 1, 3)
        layout.addWidget(self.gain_spin, 1, 4)
        layout.addWidget(self._param_label("freq"), 0, 5)
        layout.addWidget(self.freq_spin, 1, 5)
        layout.addWidget(self._param_label("q"), 0, 6)
        layout.addWidget(self.q_spin, 1, 6)

        self.enabled_check.toggled.connect(self._emit_changed)
        self.type_combo.currentTextChanged.connect(self._emit_changed)
        self.freq_spin.valueChanged.connect(self._emit_changed)
        self.q_spin.valueChanged.connect(self._emit_changed)
        self.gain_spin.valueChanged.connect(self._on_gain_spin_changed)
        self.gain_spin.editingFinished.connect(self._sync_gain_from_spin)
        self.gain_dial.valueChanged.connect(self._on_gain_dial_changed)

    def _param_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("paramLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedHeight(24)
        return label

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.selected.emit(self)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def to_filter(self) -> EqFilter:
        return EqFilter(
            self.type_combo.currentText(),
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
        self.store = PresetStore()
        self.chat_store = ChatStore()
        self.ai_service = AiEqualizerService()
        self.audio_engine = AudioEngine()
        self.current_preset = flat_preset()
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
        self.chat_messages: list[dict[str, str]] = []
        self.filter_rows: list[FilterEditorRow] = []
        self.selected_filter_row = -1
        self.settings = QSettings()
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
        self.refresh_devices()
        self.refresh_ai_models()
        self.refresh_chat_sessions()
        self.restore_chat_session()
        self.refresh_curve_lists()
        self.refresh_presets()
        self.populate_filter_editor()
        self.update_graph()
        self.restore_window_layout()
        self.schedule_filter_container_sync()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast("Дождитесь ответа ИИ-агента")
            event.ignore()
            return
        self.save_window_layout()
        self.audio_engine.stop()
        self.audio_latency_timer.stop()
        self.ai_service.shutdown()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        root.addWidget(self._build_top_bar())

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self._build_left_panel())
        self.main_splitter.addWidget(self._build_chat_panel())
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        root.addWidget(self.main_splitter, 1)
        self.setCentralWidget(central)

        self.toast_label = QLabel(central)
        self.toast_label.setObjectName("toast")
        self.toast_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.toast_label.hide()

    def restore_window_layout(self) -> None:
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        splitter_state = self.settings.value("window/main_splitter")
        if geometry:
            self.restoreGeometry(geometry)
            if state:
                self.restoreState(state)
        else:
            self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
            screen = QApplication.primaryScreen()
            if screen is not None:
                frame = self.frameGeometry()
                frame.moveCenter(screen.availableGeometry().center())
                self.move(frame.topLeft())
        if splitter_state:
            self.main_splitter.restoreState(splitter_state)
        else:
            self.main_splitter.setSizes(list(DEFAULT_SPLITTER_SIZES))
        self.clamp_window_to_available_screen()
        self.settings.setValue("window/geometry", self.saveGeometry())

    def save_window_layout(self) -> None:
        self.clamp_window_to_available_screen()
        self.settings.setValue("window/geometry", self.saveGeometry())
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

    def _build_top_bar(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.input_combo = SearchableComboBox(empty_text="Нет входов")
        self.output_combo = SearchableComboBox(empty_text="Нет выходов")
        self.sample_rate_combo = SearchableComboBox(empty_text="Нет частот")
        self.audio_dtype_combo = SearchableComboBox(empty_text="Нет форматов")
        self._configure_flexible_combo(self.input_combo, min_chars=24)
        self._configure_flexible_combo(self.output_combo, min_chars=24)
        self.sample_rate_combo.setFixedWidth(104)
        self.audio_dtype_combo.setFixedWidth(118)
        self.input_combo.currentIndexChanged.connect(self.refresh_audio_settings)
        self.output_combo.currentIndexChanged.connect(self.refresh_audio_settings)
        self.sample_rate_combo.currentIndexChanged.connect(self.refresh_audio_dtype_options)
        self.refresh_devices_button = QPushButton("↻")
        self.refresh_devices_button.setObjectName("refreshDevicesButton")
        self.refresh_devices_button.setToolTip("Обновить устройства")
        self.refresh_devices_button.setFixedSize(34, 30)
        self.refresh_devices_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.refresh_devices_button.clicked.connect(self.refresh_devices)

        self.audio_button = QPushButton("▶")
        self.audio_button.setObjectName("audioButton")
        self.audio_button.setProperty("running", False)
        self.audio_button.setToolTip("Старт")
        self.audio_button.setFixedSize(38, 30)
        self.audio_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.audio_button.clicked.connect(self.toggle_audio)
        self.status_label = QLabel("--")
        self.status_label.setObjectName("latencyLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        transport = QWidget()
        transport_layout = QHBoxLayout(transport)
        transport_layout.setContentsMargins(0, 0, 0, 0)
        transport_layout.setSpacing(4)
        transport_layout.addWidget(self.audio_button)
        transport_layout.addWidget(self.status_label)

        layout.addWidget(QLabel("Вход"), 0, 0)
        layout.addWidget(self.input_combo, 0, 1)
        layout.addWidget(QLabel("Выход"), 0, 2)
        layout.addWidget(self.output_combo, 0, 3)
        layout.addWidget(QLabel("SR (Hz)"), 0, 4)
        layout.addWidget(self.sample_rate_combo, 0, 5)
        layout.addWidget(QLabel("Формат"), 0, 6)
        layout.addWidget(self.audio_dtype_combo, 0, 7)
        layout.addWidget(self.refresh_devices_button, 0, 8)
        layout.addWidget(transport, 0, 9)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(3, 2)
        return panel

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_graph_section(), 3)
        layout.addWidget(self._build_filters_section(), 2)
        return panel

    def _build_graph_section(self) -> QGroupBox:
        box = QGroupBox("АЧХ")
        layout = QVBoxLayout(box)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(axisItems={"bottom": FrequencyAxisItem(orientation="bottom")})
        self.plot.setBackground("#111318")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLogMode(x=True, y=False)
        self.plot.setLabel("bottom", "Частота")
        self.plot.setLabel("left", "Усиление", units="dB")
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
            verSpacing=4,
            brush=pg.mkBrush(17, 19, 24, 210),
            pen=pg.mkPen("#3a414d"),
            labelTextColor="#e8edf2",
        )
        self.plot_legend.setParentItem(plot_item.vb)
        plot_item.legend = self.plot_legend
        self.device_curve_item = self.plot.plot(
            GRAPH_FREQS,
            np.zeros_like(GRAPH_FREQS),
            pen=pg.mkPen(DEVICE_CURVE_COLOR, width=2),
            name="Устройство",
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
            name="Текущий",
        )
        self.current_curve.setZValue(10)
        self.compare_curves: dict[int, pg.PlotDataItem] = {}
        layout.addWidget(self.plot, 1)

        controls = QHBoxLayout()
        self.device_curve_combo = SearchableComboBox(empty_text="Нет устройств")
        self._configure_flexible_combo(self.device_curve_combo, min_chars=14)
        self.device_curve_combo.setMaxVisibleItems(12)
        self.device_curve_combo.currentIndexChanged.connect(self.on_device_curve_changed)
        self.refresh_curves_button = QPushButton("↻")
        self.refresh_curves_button.setObjectName("refreshCurvesButton")
        self.refresh_curves_button.setToolTip("Обновить списки")
        self.refresh_curves_button.setFixedSize(34, 30)
        self.refresh_curves_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.refresh_curves_button.clicked.connect(lambda: self.refresh_curve_lists(show_feedback=True))
        self.current_selector = SearchableComboBox(empty_text="Нет пресетов")
        self._configure_flexible_combo(self.current_selector, min_chars=18)
        self.current_selector.currentIndexChanged.connect(self.load_current_from_selector)
        self.compare_button = QPushButton("Сравнить")
        self.compare_button.setObjectName("compareButton")
        self.compare_button.setFixedWidth(98)
        self.compare_button.setFixedHeight(30)
        self.compare_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.compare_button.setText("Сравнить")
        self.compare_menu = QMenu(self.compare_button)
        self.compare_button.setMenu(self.compare_menu)
        self.import_button = QPushButton("Импорт")
        self.import_button.setObjectName("miniButton")
        self.import_button.setFixedWidth(82)
        self.import_button.setFixedHeight(30)
        self.import_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.import_button.clicked.connect(self.import_preset)
        self.export_button = QPushButton("Экспорт")
        self.export_button.setObjectName("miniButton")
        self.export_button.setFixedWidth(82)
        self.export_button.setFixedHeight(30)
        self.export_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_button.clicked.connect(self.export_preset)

        controls.addWidget(QLabel("Устройство"))
        controls.addWidget(self.device_curve_combo, 1)
        controls.addWidget(self.refresh_curves_button)
        controls.addWidget(QLabel("Текущий пресет"))
        controls.addWidget(self.current_selector, 2)
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
        box = QGroupBox("Фильтры")
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
        layout.addWidget(self.filter_scroll, 1)

        buttons = QHBoxLayout()
        self.add_filter_button = QPushButton("Добавить")
        self.add_filter_button.clicked.connect(self.add_filter)
        self.remove_filter_button = QPushButton("Удалить")
        self.remove_filter_button.clicked.connect(self.remove_selected_filter)
        self.clear_filters_button = QPushButton("Очистить")
        self.clear_filters_button.clicked.connect(self.clear_filters)
        self.delete_preset_button = QPushButton("Удалить пресет")
        self.delete_preset_button.clicked.connect(self.delete_current_preset)
        self.save_button = QPushButton("Сохранить пресет")
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
        box = QGroupBox("Ассистент")
        layout = QVBoxLayout(box)
        self.side_tabs = QTabWidget()

        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        ai_layout.setContentsMargins(0, 10, 0, 0)
        ai_layout.setSpacing(8)
        model_controls = QHBoxLayout()
        model_controls.addWidget(QLabel("Модель"))
        self.ai_model_combo = SearchableComboBox(empty_text="Нет моделей")
        self.ai_model_combo.setMaxVisibleItems(12)
        self._configure_flexible_combo(self.ai_model_combo, min_chars=18)
        model_controls.addWidget(self.ai_model_combo, 1)
        self.refresh_models_button = QPushButton("↻")
        self.refresh_models_button.setToolTip("Обновить модели")
        self.refresh_models_button.setFixedSize(34, 30)
        self.refresh_models_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.refresh_models_button.clicked.connect(self.refresh_ai_models)
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
        self.chat_input.setPlaceholderText("Например: убери гул, добавь воздуха, вокал резкий")
        self.chat_input.setFixedHeight(76)
        self.chat_input.submit_requested.connect(self.send_chat)
        self.chat_menu_button = QPushButton("≡")
        self.chat_menu_button.setObjectName("chatIconButton")
        self.chat_menu_button.setToolTip("Сохраненные чаты")
        self.chat_menu_button.setFixedSize(30, 30)
        self.chat_menu = QMenu(self.chat_menu_button)
        self.chat_menu_button.clicked.connect(self.show_chat_menu)
        self.delete_chat_button = QPushButton("×")
        self.delete_chat_button.setObjectName("chatIconButton")
        self.delete_chat_button.setToolTip("Удалить текущий чат")
        self.delete_chat_button.setFixedSize(30, 30)
        self.delete_chat_button.clicked.connect(self.delete_current_chat)
        self.new_chat_button = QPushButton("+")
        self.new_chat_button.setObjectName("chatIconButton")
        self.new_chat_button.setToolTip("Новый чат")
        self.new_chat_button.setFixedSize(30, 30)
        self.new_chat_button.clicked.connect(self.start_new_chat)
        self.send_button = QPushButton("↑")
        self.send_button.setObjectName("chatIconButton")
        self.send_button.setToolTip("Отправить")
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
        self.append_chat("AIEQ", CHAT_INTRO_TEXT)

        autoeq_tab = QWidget()
        autoeq_layout = QVBoxLayout(autoeq_tab)
        autoeq_layout.setContentsMargins(0, 10, 0, 0)
        autoeq_layout.setSpacing(8)
        autoeq_layout.addWidget(QLabel("Целевая кривая"))
        self.target_curve_combo = SearchableComboBox(empty_text="Нет кривых")
        self.target_curve_combo.setMaxVisibleItems(12)
        self.target_curve_combo.currentIndexChanged.connect(self.on_target_curve_changed)
        autoeq_layout.addWidget(self.target_curve_combo)
        self.show_target_checkbox = QCheckBox("Показывать target")
        self.show_target_checkbox.toggled.connect(lambda _checked: self.update_graph())
        autoeq_layout.addWidget(self.show_target_checkbox)
        autoeq_layout.addWidget(QLabel("Алгоритм"))
        self.autoeq_backend_combo = SearchableComboBox(empty_text="Нет алгоритмов")
        self.autoeq_backend_combo.addItem("dmitryz1024", "local")
        self.autoeq_backend_combo.addItem("jaakkopasanen", "official")
        autoeq_layout.addWidget(self.autoeq_backend_combo)
        self.run_autoeq_button = QPushButton("Рассчитать AutoEQ")
        self.run_autoeq_button.clicked.connect(self.run_autoeq)
        autoeq_layout.addWidget(self.run_autoeq_button)
        autoeq_layout.addStretch(1)

        self.side_tabs.addTab(ai_tab, "AI чат")
        self.side_tabs.addTab(autoeq_tab, "AutoEQ")
        layout.addWidget(self.side_tabs, 1)
        return box

    def _apply_style(self) -> None:
        QApplication.instance().setStyleSheet(
            """
            QWidget {
                background: #181a1f;
                color: #e8edf2;
                font-size: 13px;
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
            QPushButton:disabled {
                background: #1d222b;
                border-color: #2b313b;
                color: #707987;
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
            QPushButton#compareButton::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                right: 7px;
            }
            QComboBox, QTextEdit, QTextBrowser, QScrollArea, QDoubleSpinBox {
                background: #111318;
                border: 1px solid #313640;
                border-radius: 6px;
                padding: 5px;
                selection-background-color: #05e5b6;
                outline: none;
            }
            QDoubleSpinBox {
                padding: 5px;
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
            QScrollArea#filterScroll {
                background: #181a1f;
                border: 1px solid #313640;
                border-radius: 6px;
                padding: 0;
            }
            QWidget#filterViewport,
            QWidget#filterContainer {
                background: #181a1f;
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
                padding: 2px 6px;
                font-weight: 500;
            }
            QDial {
                background: transparent;
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
            }
            QLineEdit#comboSearch, QListWidget#comboSearchList {
                background: #111318;
                border: 1px solid #313640;
                border-radius: 6px;
                padding: 5px;
                selection-background-color: #05e5b6;
                outline: none;
            }
            QListWidget#comboSearchList::item {
                padding: 6px 8px;
            }
            QListWidget#comboSearchList::item:selected {
                background: #2d3440;
            }
            QListWidget#comboSearchList::item:disabled {
                color: #707987;
            }
            """
        )

    def refresh_devices(self) -> None:
        if self.audio_engine.is_running:
            self.refresh_devices_button.setEnabled(False)
            return
        self.refresh_devices_button.setEnabled(True)
        previous_input = self.input_combo.currentData()
        previous_output = self.output_combo.currentData()
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
            self.show_toast("sounddevice недоступен")
            self.audio_button.setEnabled(False)
            self.refresh_devices_button.setEnabled(True)
            return

        for device in self.input_devices:
            self.input_combo.addItem(device.label, device.index)
        for device in self.output_devices:
            self.output_combo.addItem(device.label, device.index)

        self._restore_combo_data(self.input_combo, previous_input)
        self._restore_combo_data(self.output_combo, previous_output)
        self.input_combo.blockSignals(False)
        self.output_combo.blockSignals(False)
        self.refresh_audio_settings()

        self.audio_button.setEnabled(bool(self.input_devices and self.output_devices))
        self.refresh_devices_button.setEnabled(True)
        self.status_label.setToolTip("")
        self.update_audio_latency_label()

    def _restore_combo_data(self, combo: QComboBox, value: object) -> None:
        if value is None:
            return
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

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

    def refresh_ai_models(self) -> None:
        if not hasattr(self, "ai_model_combo"):
            return
        previous = self.ai_model_combo.currentData()
        default_path = str(self.ai_service.llama_model_path)
        self.ai_model_combo.blockSignals(True)
        self.ai_model_combo.clear()
        models = list_local_models()
        if not models:
            self.ai_model_combo.addItem("Модели не найдены", None)
            self.ai_model_combo.setEnabled(False)
            self.ai_model_combo.blockSignals(False)
            return
        self.ai_model_combo.setEnabled(True)
        selected_index = 0
        wanted = str(previous) if previous else default_path
        for index, model_path in enumerate(models):
            resolved = str(model_path)
            self.ai_model_combo.addItem(model_path.name, resolved)
            if resolved == wanted or str(model_path.resolve()) == wanted:
                selected_index = index
        self.ai_model_combo.setCurrentIndex(selected_index)
        self.ai_model_combo.blockSignals(False)

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
        self.device_curves = list_curves(DEVICE_CURVES_DIR, include_default=True)
        self.target_curves = list_curves(TARGET_CURVES_DIR, include_default=False)

        self.device_curve_combo.blockSignals(True)
        self.device_curve_combo.clear()
        device_index = 0
        for index, curve in enumerate(self.device_curves):
            self.device_curve_combo.addItem(curve.name, index)
            if curve.name == previous_device:
                device_index = index
        self.device_curve_combo.setCurrentIndex(device_index)
        self.device_curve_combo.blockSignals(False)
        self.selected_device_curve = self.device_curves[device_index] if self.device_curves else None

        self.target_curve_combo.blockSignals(True)
        self.target_curve_combo.clear()
        self.device_curve_combo.setEnabled(bool(self.device_curves))
        self.rebuild_target_curve_combo(previous_target)
        self.update_graph()
        if show_feedback:
            self.show_toast("Списки кривых обновлены")

    def on_device_curve_changed(self, index: int) -> None:
        if 0 <= index < len(self.device_curves):
            previous_target = self.target_curve_combo.currentText()
            self.selected_device_curve = self.device_curves[index]
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
        return self.target_options[int(target_index)].response_db(GRAPH_FREQS)

    def refresh_presets(self) -> None:
        self.saved_presets = self.store.list_presets()
        self._updating = True
        self.current_selector.clear()
        self.current_selector.addItem("New", NEW_PRESET_ID)
        selected_index = 0
        for preset in self.saved_presets:
            self.current_selector.addItem(preset.name, preset.id)
            if preset.id is not None and preset.id == self.current_preset.id:
                selected_index = self.current_selector.count() - 1
        self.current_selector.setCurrentIndex(selected_index)
        self._updating = False
        self.rebuild_compare_menu()
        self.update_graph()

    def rebuild_compare_menu(self) -> None:
        self.compare_menu.clear()
        current_id = self.current_preset.id
        for idx, preset in enumerate(self.saved_presets):
            if preset.id is None or preset.id == current_id:
                continue
            action = QAction(preset.name, self.compare_menu)
            action.setCheckable(True)
            action.setChecked(preset.id in self.compare_ids)
            color = CURVE_COLORS[idx % len(CURVE_COLORS)]
            action.setData((preset.id, color))
            action.toggled.connect(self.on_compare_toggled)
            self.compare_menu.addAction(action)
        if not self.compare_menu.actions():
            action = QAction("Нет сохраненных пресетов", self.compare_menu)
            action.setEnabled(False)
            self.compare_menu.addAction(action)

    def on_compare_toggled(self, checked: bool) -> None:
        action = self.sender()
        if not isinstance(action, QAction):
            return
        preset_id, _color = action.data()
        if checked:
            self.compare_ids.add(int(preset_id))
        else:
            self.compare_ids.discard(int(preset_id))
        self.update_graph()

    def update_graph(self) -> None:
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

        for idx, preset in enumerate(self.saved_presets):
            if preset.id is None or preset.id not in self.compare_ids or preset.id == self.current_preset.id:
                continue
            color = CURVE_COLORS[idx % len(CURVE_COLORS)]
            curve = self.plot.plot(
                GRAPH_FREQS,
                device_db + preset_response_db(preset, GRAPH_FREQS, DEFAULT_SAMPLE_RATE),
                pen=pg.mkPen(color, width=2),
                name=preset.name,
            )
            self.compare_curves[preset.id] = curve

    def set_target_legend_visible(self, visible: bool) -> None:
        if not hasattr(self, "plot_legend"):
            return
        if visible == self.target_legend_visible:
            return
        if visible:
            self.plot_legend.addItem(self.target_curve_item, "Target")
        else:
            self.plot_legend.removeItem("Target")
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
        self.filter_container.setFixedSize(width, height)
        self.filter_container.updateGeometry()
        self.filter_container.update()
        self.filter_scroll.viewport().update()

    def _add_filter_row(self, eq_filter: EqFilter, index: int) -> None:
        row = FilterEditorRow(eq_filter, index)
        row.installEventFilter(self)
        row.changed.connect(self.on_filters_changed)
        row.selected.connect(self.select_filter_row)
        self.filter_rows.append(row)
        self.filter_list_layout.insertWidget(
            self.filter_list_layout.count() - 1,
            row,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )

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
        if self.current_preset.id is not None:
            self.current_preset = self.current_preset.clone(name=f"{self.current_preset.name} (ред.)", keep_id=False)
            self.refresh_presets()
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

    def has_unsaved_changes(self) -> bool:
        if self.current_preset.id is not None:
            return False
        return bool(self.current_preset.filters) or self.current_preset.name != "New"

    def confirm_or_save_before_switch(self) -> bool:
        if not self.has_unsaved_changes():
            return True
        answer = QMessageBox.question(
            self,
            "Сохранить изменения",
            "Хотите сохранить, чтобы зафиксировать изменения?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            return self.save_current_preset()
        return True

    def new_flat_preset(self) -> None:
        self.current_preset = flat_preset()
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
        self.populate_filter_editor()
        if isinstance(preset_id, int):
            self.compare_ids.discard(int(preset_id))
        self.rebuild_compare_menu()
        self.refresh_presets()
        self.update_graph()
        self.apply_audio_preset()

    def delete_current_preset(self) -> None:
        if self.current_preset.id is None:
            self.show_toast("Текущий пресет еще не сохранен")
            return
        preset_id = self.current_preset.id
        answer = QMessageBox.question(
            self,
            "Удалить пресет",
            f"Удалить пресет «{self.current_preset.name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.store.delete(preset_id)
        self.compare_ids.discard(preset_id)
        self.current_preset = flat_preset()
        self.populate_filter_editor()
        self.refresh_presets()
        self.apply_audio_preset()
        self.show_toast("Пресет удален")

    def save_current_preset(self) -> bool:
        name, accepted = QInputDialog.getText(self, "Сохранить пресет", "Название пресета", text=self.current_preset.name)
        if not accepted:
            return False
        name = name.strip() or self.current_preset.name
        if self.is_reserved_preset_name(name):
            QMessageBox.warning(self, "Сохранить пресет", "Название New зарезервировано для нового пресета.")
            return False
        existing = self.store.get_preset_by_name(name)
        if existing is not None:
            answer = QMessageBox.question(
                self,
                "Перезаписать пресет",
                f"Пресет «{name}» уже существует. Перезаписать его?",
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
        self.refresh_presets()
        self.show_toast("Пресет сохранен")
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
        path, _ = QFileDialog.getOpenFileName(self, "Импорт пресета", str(Path.cwd()), "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.current_preset = Preset.from_dict(data)
            self.populate_filter_editor()
            self.refresh_presets()
            self.apply_audio_preset()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Импорт", f"Не удалось импортировать пресет:\n{exc}")

    def export_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт пресета", f"{self.current_preset.name}.json", "JSON (*.json)")
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        try:
            target.write_text(json.dumps(self.current_preset.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Экспорт", f"Не удалось экспортировать пресет:\n{exc}")

    def run_autoeq(self) -> None:
        if self.selected_device_curve is None:
            self.show_toast("Выберите устройство")
            return
        target_index = self.target_curve_combo.currentData()
        if target_index is None or not (0 <= int(target_index) < len(self.target_options)):
            self.show_toast("Выберите целевую кривую")
            return
        target_curve = self.target_options[int(target_index)]
        backend = str(self.autoeq_backend_combo.currentData() or "local")
        try:
            result = build_autoeq_preset_result(self.selected_device_curve, target_curve, backend=backend)
        except AutoEqOfficialUnavailable:
            self.show_toast("AutoEQ недоступен", timeout_ms=4200)
            return
        preset = result.preset
        preset_name = self.autoeq_preset_name(
            self.autoeq_backend_combo.currentText(),
            self.selected_device_curve.name,
            target_curve.name,
        )
        saved = self.save_generated_preset(preset, name=preset_name)
        self.current_preset = saved.clone(keep_id=True)
        self.show_target_checkbox.setChecked(True)
        self.populate_filter_editor()
        self.refresh_presets()
        self.apply_audio_preset()
        self.show_toast("AutoEQ применен")

    def set_audio_running_ui(self, running: bool) -> None:
        self.audio_button.setText("■" if running else "▶")
        self.audio_button.setToolTip("Стоп" if running else "Старт")
        self.audio_button.setProperty("running", running)
        self.audio_button.style().unpolish(self.audio_button)
        self.audio_button.style().polish(self.audio_button)
        self.audio_button.update()
        self.refresh_devices_button.setEnabled(not running)
        self.input_combo.setEnabled(not running)
        self.output_combo.setEnabled(not running)
        self.sample_rate_combo.setEnabled(not running and self.sample_rate_combo.count() > 0)
        self.audio_dtype_combo.setEnabled(not running and self.audio_dtype_combo.count() > 0)

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

        input_device = self._selected_device(self.input_combo, self.input_devices)
        output_device = self._selected_device(self.output_combo, self.output_devices)
        if input_device is None or output_device is None:
            QMessageBox.warning(self, "Аудио", "Выберите вход и выход.")
            return
        try:
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
            QMessageBox.critical(self, "Аудио", f"Не удалось запустить поток:\n{exc}")

    def _selected_device(self, combo: QComboBox, devices: list[AudioDevice]) -> AudioDevice | None:
        index = combo.currentData()
        for device in devices:
            if device.index == index:
                return device
        return None

    def append_chat(self, author: str, text: str) -> None:
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        color = CURRENT_COLOR if author != "Вы" else USER_CHAT_COLOR
        self.chat_history.append(f'<p><b style="color:{color}">{author}</b><br>{safe}</p>')

    def show_chat_menu(self) -> None:
        self.refresh_chat_sessions()
        self.chat_menu.exec(self.chat_menu_button.mapToGlobal(self.chat_menu_button.rect().bottomLeft()))

    def refresh_chat_sessions(self) -> None:
        if not hasattr(self, "chat_menu"):
            return
        self.chat_sessions = self.chat_store.list_sessions()
        self.chat_menu.clear()
        if not self.chat_sessions:
            action = QAction("Сохраненных чатов нет", self.chat_menu)
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
            self.show_toast("ИИ-агент еще отвечает")
            return
        if chat_id is None:
            self.start_new_chat(show_feedback=show_feedback)
            return
        session = self.chat_store.get_session(chat_id)
        if session is None:
            self.show_toast("Чат не найден")
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
            self.show_toast("Чат открыт")

    def start_new_chat(self, *, show_feedback: bool = True) -> None:
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast("ИИ-агент еще отвечает")
            return
        self.ai_service.clear_context()
        self.current_chat_id = None
        self.current_chat_context_full = False
        self.chat_messages = []
        self.settings.remove("ai/current_chat_id")
        self.render_chat_messages()
        self.refresh_chat_sessions()
        if show_feedback:
            self.show_toast("Новый чат")

    def delete_current_chat(self) -> None:
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast("ИИ-агент еще отвечает")
            return
        if self.current_chat_id is None:
            self.start_new_chat()
            return
        answer = QMessageBox.question(
            self,
            "Удалить чат",
            "Удалить текущий чат?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.chat_store.delete(self.current_chat_id)
        self.show_toast("Чат удален")
        self.start_new_chat(show_feedback=False)

    def render_chat_messages(self) -> None:
        self.chat_history.clear()
        self.append_chat("AIEQ", CHAT_INTRO_TEXT)
        for message in self.chat_messages:
            author = "Вы" if message.get("role") == "user" else "AIEQ"
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
        self.show_toast("Контекст чата заполнен", timeout_ms=4200)

    def update_chat_context_state(self) -> None:
        running = self._ai_thread is not None and self._ai_thread.isRunning()
        blocked = self.current_chat_context_full
        self.chat_input.setEnabled(not blocked)
        self.send_button.setEnabled(not blocked and not running)
        if blocked:
            self.chat_input.setPlaceholderText("Контекст этого чата заполнен. Создайте новый чат или откройте другой.")
        else:
            self.chat_input.setPlaceholderText("Например: убери гул, добавь воздуха, вокал резкий")

    def show_toast(self, text: str, timeout_ms: int = 2200) -> None:
        self.toast_label.setText(text)
        self.toast_label.setMinimumWidth(0)
        self.toast_label.setMaximumWidth(16777215)
        self.toast_label.adjustSize()
        width = min(max(self.toast_label.sizeHint().width(), 180), max(180, self.width() - 80))
        self.toast_label.setFixedWidth(width)
        self.toast_label.adjustSize()
        self._position_toast()
        self.toast_label.show()
        self.toast_timer.start(timeout_ms)

    def hide_toast(self) -> None:
        self.toast_label.hide()

    def _position_toast(self) -> None:
        if not hasattr(self, "toast_label"):
            return
        central = self.centralWidget()
        width = central.width() if central is not None else self.width()
        x = max(20, (width - self.toast_label.width()) // 2)
        top = self.main_splitter.geometry().top() if hasattr(self, "main_splitter") else 20
        y = max(14, top + 4)
        self.toast_label.move(x, y)

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
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
        self.sync_filter_container_width()
        self.schedule_filter_container_sync()
        self._position_toast()

    def send_chat(self) -> None:
        text = self.chat_input.toPlainText().strip()
        if not text:
            return
        if self.current_chat_context_full:
            self.show_toast("Контекст чата заполнен", timeout_ms=4200)
            self.update_chat_context_state()
            return
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast("ИИ-агент еще отвечает")
            return
        history_for_model = list(self.chat_messages)
        self.ensure_current_chat(text)
        self.chat_input.clear()
        self.send_button.setEnabled(False)
        self.append_chat("Вы", text)
        self.chat_messages.append({"role": "user", "content": text})
        self.save_current_chat()
        self.show_toast("ИИ-агент думает")

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
        self.populate_filter_editor()
        self.refresh_presets()
        self.append_chat("AIEQ", result.assistant_message)
        self.chat_messages.append({"role": "assistant", "content": result.assistant_message})
        self.show_toast("Пресет применен")
        self.save_current_chat()
        self.send_button.setEnabled(not self.current_chat_context_full)
        self.apply_audio_preset()

    def on_ai_thread_finished(self) -> None:
        self._ai_thread = None
        self._ai_worker = None
        self.update_chat_context_state()
