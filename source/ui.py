from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
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
    QVBoxLayout,
    QWidget,
)

from .ai import AiEqualizerService, AiPresetResult
from .autoeq_service import build_autoeq_preset
from .audio import AudioDevice, AudioEngine, list_audio_devices
from .curves import DEVICE_CURVES_DIR, TARGET_CURVES_DIR, FrequencyCurve, ensure_curve_dirs, list_curves
from .dsp import DEFAULT_SAMPLE_RATE, GRAPH_FREQS, preset_response_db
from .models import FILTER_TYPES, EqFilter, Preset, flat_preset
from .storage import PresetStore

CURVE_COLORS = [
    "#16c7b7",
    "#f2b84b",
    "#7ddc63",
    "#5aa9ff",
    "#d984ff",
    "#ff8a4c",
    "#9ad7ff",
]
CURRENT_COLOR = "#ff3f6e"
NEW_PRESET_ID = "__new__"
DEFAULT_WINDOW_WIDTH = 1480
DEFAULT_WINDOW_HEIGHT = 900
DEFAULT_SPLITTER_SIZES = (1040, 440)


class FrequencyAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):  # type: ignore[override]
        labels: list[str] = []
        for value in values:
            freq = 10.0**value
            if freq >= 1000.0:
                shown = freq / 1000.0
                labels.append(f"{shown:g}k")
            else:
                labels.append(f"{freq:.0f}")
        return labels


class AiWorker(QObject):
    finished = Signal(object)

    def __init__(self, service: AiEqualizerService, text: str, preset: Preset) -> None:
        super().__init__()
        self.service = service
        self.text = text
        self.preset = preset

    def run(self) -> None:
        self.finished.emit(self.service.suggest_preset(self.text, self.preset))


class FilterEditorRow(QFrame):
    changed = Signal()
    selected = Signal(object)

    def __init__(self, eq_filter: EqFilter, index: int) -> None:
        super().__init__()
        self.index = index
        self._syncing = False
        self.setObjectName("filterRow")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(450)
        self.setMinimumHeight(122)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(5)

        self.enabled_check = QCheckBox()
        self.enabled_check.setToolTip("Вкл")
        self.enabled_check.setChecked(eq_filter.enabled)
        self.type_combo = QComboBox()
        self.type_combo.addItems(FILTER_TYPES)
        self.type_combo.setCurrentText(eq_filter.type)
        self.type_combo.setFixedWidth(84)

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
        self.gain_spin.setFixedWidth(72)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20.0, 20000.0)
        self.freq_spin.setDecimals(0)
        self.freq_spin.setSingleStep(10.0)
        self.freq_spin.setKeyboardTracking(False)
        self.freq_spin.setValue(eq_filter.freq)
        self.freq_spin.setFixedWidth(82)

        self.q_spin = QDoubleSpinBox()
        self.q_spin.setRange(0.1, 18.0)
        self.q_spin.setDecimals(3)
        self.q_spin.setSingleStep(0.01)
        self.q_spin.setKeyboardTracking(False)
        self.q_spin.setValue(eq_filter.q)
        self.q_spin.setFixedWidth(70)

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
        self.ai_service = AiEqualizerService()
        self.audio_engine = AudioEngine()
        self.current_preset = flat_preset()
        self.saved_presets: list[Preset] = []
        self.compare_ids: set[int] = set()
        self.device_curves: list[FrequencyCurve] = []
        self.target_curves: list[FrequencyCurve] = []
        self.selected_device_curve: FrequencyCurve | None = None
        self.input_devices: list[AudioDevice] = []
        self.output_devices: list[AudioDevice] = []
        self._updating = False
        self._ai_thread: QThread | None = None
        self._ai_worker: AiWorker | None = None
        self.filter_rows: list[FilterEditorRow] = []
        self.selected_filter_row = -1
        self.settings = QSettings()
        self.audio_update_timer = QTimer(self)
        self.audio_update_timer.setSingleShot(True)
        self.audio_update_timer.timeout.connect(self.apply_audio_preset)
        self.toast_timer = QTimer(self)
        self.toast_timer.setSingleShot(True)
        self.toast_timer.timeout.connect(self.hide_toast)

        ensure_curve_dirs()
        self._build_ui()
        self._apply_style()
        self.refresh_devices()
        self.refresh_curve_lists()
        self.refresh_presets()
        self.populate_filter_editor()
        self.update_graph()
        self.restore_window_layout()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast("Дождитесь ответа ИИ-агента")
            event.ignore()
            return
        self.save_window_layout()
        self.audio_engine.stop()
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

    def save_window_layout(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("window/main_splitter", self.main_splitter.saveState())

    def _build_top_bar(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.refresh_devices_button = QPushButton("Обновить")
        self.refresh_devices_button.clicked.connect(self.refresh_devices)

        self.audio_button = QPushButton("Старт")
        self.audio_button.clicked.connect(self.toggle_audio)
        self.status_label = QLabel("Аудио остановлено")

        layout.addWidget(QLabel("Вход"), 0, 0)
        layout.addWidget(self.input_combo, 0, 1)
        layout.addWidget(QLabel("Выход"), 0, 2)
        layout.addWidget(self.output_combo, 0, 3)
        layout.addWidget(self.refresh_devices_button, 0, 4)
        layout.addWidget(self.audio_button, 0, 5)
        layout.addWidget(self.status_label, 0, 6)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(3, 2)
        layout.setColumnStretch(6, 1)
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
        self.plot.addLegend(offset=(12, 12))
        self.device_curve_item = self.plot.plot(
            GRAPH_FREQS,
            np.zeros_like(GRAPH_FREQS),
            pen=pg.mkPen("#2a2d33", width=2),
            name="Устройство",
        )
        self.device_curve_item.setZValue(-10)
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
        self.device_curve_combo = QComboBox()
        self.device_curve_combo.setFixedWidth(180)
        self.device_curve_combo.setMaxVisibleItems(12)
        self.device_curve_combo.currentIndexChanged.connect(self.on_device_curve_changed)
        self.current_selector = QComboBox()
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
        controls.addWidget(self.device_curve_combo)
        controls.addWidget(QLabel("Текущий пресет"))
        controls.addWidget(self.current_selector, 2)
        controls.addWidget(self.compare_button)
        controls.addWidget(self.import_button)
        controls.addWidget(self.export_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        return box

    def _build_filters_section(self) -> QGroupBox:
        box = QGroupBox("Фильтры")
        layout = QVBoxLayout(box)

        self.filter_scroll = QScrollArea()
        self.filter_scroll.setWidgetResizable(True)
        self.filter_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.filter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.filter_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.filter_scroll.setMinimumHeight(150)
        self.filter_container = QWidget()
        self.filter_list_layout = QHBoxLayout(self.filter_container)
        self.filter_list_layout.setContentsMargins(0, 0, 0, 8)
        self.filter_list_layout.setSpacing(8)
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
        ai_layout.setContentsMargins(0, 0, 0, 0)
        ai_layout.setSpacing(8)
        self.chat_history = QTextBrowser()
        self.chat_history.setOpenExternalLinks(True)
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Например: убери гул, добавь воздуха, вокал резкий")
        self.chat_input.setFixedHeight(96)
        self.send_button = QPushButton("Отправить")
        self.send_button.clicked.connect(self.send_chat)
        ai_layout.addWidget(self.chat_history, 1)
        ai_layout.addWidget(self.chat_input)
        ai_layout.addWidget(self.send_button)
        self.append_chat("AIEQ", "Опиши, что хочется изменить в звуке. Я сохраню ответ как новый пресет и применю его.")

        autoeq_tab = QWidget()
        autoeq_layout = QVBoxLayout(autoeq_tab)
        autoeq_layout.setContentsMargins(0, 0, 0, 0)
        autoeq_layout.setSpacing(8)
        autoeq_layout.addWidget(QLabel("Целевая кривая"))
        self.target_curve_combo = QComboBox()
        self.target_curve_combo.setMaxVisibleItems(12)
        autoeq_layout.addWidget(self.target_curve_combo)
        self.refresh_curves_button = QPushButton("Обновить списки")
        self.refresh_curves_button.clicked.connect(self.refresh_curve_lists)
        autoeq_layout.addWidget(self.refresh_curves_button)
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
            }
            QPushButton:hover {
                background: #2d3440;
                border-color: #586171;
            }
            QPushButton:pressed {
                background: #1f242c;
            }
            QPushButton#miniButton, QPushButton#compareButton {
                padding: 4px 8px;
                font-size: 12px;
                min-height: 20px;
                max-height: 30px;
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
                selection-background-color: #ff3f6e;
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
                border: 1px solid #ff3f6e;
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
            """
        )

    def refresh_devices(self) -> None:
        self.input_combo.clear()
        self.output_combo.clear()
        try:
            self.input_devices = list_audio_devices("input")
            self.output_devices = list_audio_devices("output")
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"sounddevice недоступен: {exc}")
            self.audio_button.setEnabled(False)
            return

        for device in self.input_devices:
            self.input_combo.addItem(device.label, device.index)
        for device in self.output_devices:
            self.output_combo.addItem(device.label, device.index)

        self.audio_button.setEnabled(bool(self.input_devices and self.output_devices))
        self.status_label.setText("Аудио остановлено")

    def refresh_curve_lists(self) -> None:
        previous_device = self.selected_device_curve.name if self.selected_device_curve is not None else "Default"
        previous_target = self.target_curve_combo.currentText() if hasattr(self, "target_curve_combo") else ""

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
        for index, curve in enumerate(self.target_curves):
            self.target_curve_combo.addItem(curve.name, index)
            if curve.name == previous_target:
                self.target_curve_combo.setCurrentIndex(index)
        self.target_curve_combo.blockSignals(False)
        self.run_autoeq_button.setEnabled(bool(self.target_curves))
        self.update_graph()

    def on_device_curve_changed(self, index: int) -> None:
        if 0 <= index < len(self.device_curves):
            self.selected_device_curve = self.device_curves[index]
            self.update_graph()

    def selected_device_response_db(self) -> np.ndarray:
        if self.selected_device_curve is None:
            return np.zeros_like(GRAPH_FREQS, dtype=np.float64)
        return self.selected_device_curve.response_db(GRAPH_FREQS)

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
        db = device_db + preset_response_db(self.current_preset, GRAPH_FREQS, DEFAULT_SAMPLE_RATE)
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
        row_width = 458
        self.filter_container.setMinimumWidth(max(self.filter_scroll.viewport().width(), len(self.filter_rows) * row_width))
        self.update_filter_selection()
        self._updating = False

    def _add_filter_row(self, eq_filter: EqFilter, index: int) -> None:
        row = FilterEditorRow(eq_filter, index)
        row.changed.connect(self.on_filters_changed)
        row.selected.connect(self.select_filter_row)
        self.filter_rows.append(row)
        self.filter_list_layout.insertWidget(self.filter_list_layout.count() - 1, row)

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
        self.show_toast(f"Пресет сохранен: {saved.name}")
        return True

    def next_available_preset_name(self, name: str) -> str:
        base = name.strip() or "Preset"
        existing = {preset.name.casefold() for preset in self.store.list_presets()}
        if base.casefold() not in existing:
            return base
        index = 2
        while f"{base} {index}".casefold() in existing:
            index += 1
        return f"{base} {index}"

    def save_generated_preset(self, preset: Preset) -> Preset:
        return self.store.save_new(preset, name=self.next_available_preset_name(preset.name))

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
        if target_index is None or not (0 <= int(target_index) < len(self.target_curves)):
            self.show_toast("Выберите целевую кривую")
            return
        target_curve = self.target_curves[int(target_index)]
        preset = build_autoeq_preset(self.selected_device_curve, target_curve)
        saved = self.save_generated_preset(preset)
        self.current_preset = saved.clone(keep_id=True)
        self.populate_filter_editor()
        self.refresh_presets()
        self.apply_audio_preset()
        self.show_toast(f"AutoEQ применен: {saved.name}")

    def toggle_audio(self) -> None:
        if self.audio_engine.is_running:
            self.audio_engine.stop()
            self.audio_button.setText("Старт")
            self.status_label.setText("Аудио остановлено")
            return

        input_device = self._selected_device(self.input_combo, self.input_devices)
        output_device = self._selected_device(self.output_combo, self.output_devices)
        if input_device is None or output_device is None:
            QMessageBox.warning(self, "Аудио", "Выберите вход и выход.")
            return
        try:
            self.audio_engine.start(input_device, output_device, self.current_preset)
            self.audio_button.setText("Стоп")
            self.status_label.setText(f"{int(self.audio_engine.sample_rate)} Hz, block {self.audio_engine.block_size}")
        except Exception as exc:  # noqa: BLE001
            self.audio_engine.stop()
            QMessageBox.critical(self, "Аудио", f"Не удалось запустить поток:\n{exc}")

    def _selected_device(self, combo: QComboBox, devices: list[AudioDevice]) -> AudioDevice | None:
        index = combo.currentData()
        for device in devices:
            if device.index == index:
                return device
        return None

    def append_chat(self, author: str, text: str) -> None:
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        color = CURRENT_COLOR if author != "Вы" else "#16c7b7"
        self.chat_history.append(f'<p><b style="color:{color}">{author}</b><br>{safe}</p>')

    def show_toast(self, text: str, timeout_ms: int = 2200) -> None:
        self.toast_label.setText(text)
        self.toast_label.adjustSize()
        width = min(max(self.toast_label.width(), 180), max(180, self.width() - 80))
        self.toast_label.setFixedWidth(width)
        self._position_toast()
        self.toast_label.show()
        self.toast_timer.start(timeout_ms)

    def hide_toast(self) -> None:
        self.toast_label.hide()

    def _position_toast(self) -> None:
        if not hasattr(self, "toast_label"):
            return
        x = max(20, self.width() - self.toast_label.width() - 28)
        y = max(20, self.height() - self.toast_label.height() - 28)
        self.toast_label.move(x, y)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._position_toast()

    def send_chat(self) -> None:
        text = self.chat_input.toPlainText().strip()
        if not text:
            return
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self.show_toast("ИИ-агент еще отвечает")
            return
        self.chat_input.clear()
        self.send_button.setEnabled(False)
        self.send_button.setText("Думаю...")
        self.append_chat("Вы", text)
        self.show_toast("ИИ-агент думает")

        self._ai_thread = QThread(self)
        self._ai_worker = AiWorker(self.ai_service, text, self.current_preset.clone(keep_id=True))
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
            self.send_button.setEnabled(True)
            self.send_button.setText("Отправить")
            return
        saved = self.save_generated_preset(result.preset)
        self.current_preset = saved.clone(keep_id=True)
        self.populate_filter_editor()
        self.refresh_presets()
        self.append_chat("AIEQ", result.assistant_message)
        self.show_toast(f"Применен пресет: {saved.name}")
        self.send_button.setEnabled(True)
        self.send_button.setText("Отправить")
        self.apply_audio_preset()

    def on_ai_thread_finished(self) -> None:
        self._ai_thread = None
        self._ai_worker = None
