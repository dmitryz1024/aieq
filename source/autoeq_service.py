from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .curves import FrequencyCurve
from .dsp import DEFAULT_SAMPLE_RATE, GRAPH_FREQS, filter_response_db
from .models import EqFilter, Preset


class AutoEqOfficialUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AutoEqPresetResult:
    preset: Preset
    backend: str
    warning: str | None = None


def _smooth_log_curve(values: np.ndarray, window: int = 41) -> np.ndarray:
    if values.size < window:
        return values.copy()
    kernel = np.hanning(window)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _quality_from_width(freqs: np.ndarray, residual: np.ndarray, index: int) -> float:
    peak = residual[index]
    threshold = abs(peak) * 0.42
    left = index
    right = index
    while left > 0 and abs(residual[left]) >= threshold and np.sign(residual[left]) == np.sign(peak):
        left -= 1
    while right < residual.size - 1 and abs(residual[right]) >= threshold and np.sign(residual[right]) == np.sign(peak):
        right += 1
    bandwidth = max(freqs[right] - freqs[left], freqs[index] * 0.16)
    return float(np.clip(freqs[index] / bandwidth, 0.45, 6.0))


def build_autoeq_preset_result(
    device_curve: FrequencyCurve,
    target_curve: FrequencyCurve,
    *,
    max_filters: int = 8,
) -> AutoEqPresetResult:
    backend = os.environ.get("AIEQ_AUTOEQ_BACKEND", "auto").strip().lower() or "auto"
    if backend in {"auto", "official"}:
        try:
            return AutoEqPresetResult(
                preset=_build_official_autoeq_preset(device_curve, target_curve),
                backend="official",
            )
        except Exception as exc:  # noqa: BLE001 - official package is optional.
            if backend == "official":
                raise AutoEqOfficialUnavailable(str(exc)) from exc
            warning = str(exc)
            return AutoEqPresetResult(
                preset=_build_local_autoeq_preset(device_curve, target_curve, max_filters=max_filters),
                backend="local",
                warning=warning,
            )

    return AutoEqPresetResult(
        preset=_build_local_autoeq_preset(device_curve, target_curve, max_filters=max_filters),
        backend="local",
    )


def build_autoeq_preset(device_curve: FrequencyCurve, target_curve: FrequencyCurve, *, max_filters: int = 8) -> Preset:
    return build_autoeq_preset_result(device_curve, target_curve, max_filters=max_filters).preset


def _build_official_autoeq_preset(device_curve: FrequencyCurve, target_curve: FrequencyCurve) -> Preset:
    if sys.version_info >= (3, 12):
        raise AutoEqOfficialUnavailable("official AutoEq package supports Python >=3.8,<3.12")

    _ensure_matplotlib_config_dir()

    try:
        from autoeq.constants import PEQ_CONFIGS
        from autoeq.frequency_response import FrequencyResponse
    except ImportError as exc:
        raise AutoEqOfficialUnavailable("official AutoEq package is not installed") from exc

    config_name = os.environ.get("AIEQ_AUTOEQ_CONFIG", "8_PEAKING_WITH_SHELVES")
    config = _get_official_peq_config(PEQ_CONFIGS, config_name)

    measurement = FrequencyResponse(
        name=device_curve.name,
        frequency=np.asarray(device_curve.freqs, dtype=np.float64),
        raw=np.asarray(device_curve.db, dtype=np.float64),
    )
    target = FrequencyResponse(
        name=target_curve.name,
        frequency=np.asarray(target_curve.freqs, dtype=np.float64),
        raw=np.asarray(target_curve.db, dtype=np.float64),
    )
    measurement.interpolate()
    measurement.center()
    target.interpolate()
    target.center()
    measurement.compensate(target, min_mean_error=True)
    measurement.smoothen()
    measurement.equalize(concha_interference=False)
    peqs = measurement.optimize_parametric_eq(config, int(DEFAULT_SAMPLE_RATE))

    filters: list[EqFilter] = []
    for peq in peqs:
        for official_filter in peq.filters:
            eq_filter = _official_filter_to_eq_filter(official_filter)
            if eq_filter is not None and abs(eq_filter.gain) >= 0.05:
                filters.append(eq_filter)
    filters.sort(key=lambda item: item.freq)

    return Preset(name=_autoeq_name(device_curve, target_curve), filters=filters).sanitized()


def _ensure_matplotlib_config_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    config_dir = base / "AIEQ" / "matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)


def _get_official_peq_config(configs: Any, config_name: str) -> Any:
    if config_name in configs:
        return configs[config_name]
    parts = [part.strip() for part in config_name.split(",") if part.strip()]
    if len(parts) > 1 and all(part in configs for part in parts):
        return [configs[part] for part in parts]
    available = ", ".join(sorted(str(key) for key in configs.keys())[:12])
    raise AutoEqOfficialUnavailable(f"AutoEq PEQ config not found: {config_name}. Available examples: {available}")


def _official_filter_to_eq_filter(official_filter: Any) -> EqFilter | None:
    class_name = official_filter.__class__.__name__.lower()
    if "lowshelf" in class_name or class_name == "low_shelf":
        filter_type = "low_shelf"
    elif "highshelf" in class_name or class_name == "high_shelf":
        filter_type = "high_shelf"
    elif "peaking" in class_name or class_name == "peak":
        filter_type = "peaking"
    else:
        return None
    return EqFilter(
        filter_type,
        float(getattr(official_filter, "fc")),
        float(getattr(official_filter, "q")),
        float(getattr(official_filter, "gain")),
    ).sanitized()


def _build_local_autoeq_preset(device_curve: FrequencyCurve, target_curve: FrequencyCurve, *, max_filters: int = 8) -> Preset:
    device = device_curve.response_db(GRAPH_FREQS)
    target = target_curve.response_db(GRAPH_FREQS)
    residual = _smooth_log_curve(target - device)
    filters: list[EqFilter] = []

    low_mask = (GRAPH_FREQS >= 30.0) & (GRAPH_FREQS <= 140.0)
    high_mask = (GRAPH_FREQS >= 9000.0) & (GRAPH_FREQS <= 18000.0)

    low_gain = float(np.clip(np.mean(residual[low_mask]), -9.0, 9.0))
    if abs(low_gain) >= 1.0:
        eq_filter = EqFilter("low_shelf", 105.0, 0.7, low_gain)
        filters.append(eq_filter)
        residual -= filter_response_db(eq_filter, GRAPH_FREQS)

    high_gain = float(np.clip(np.mean(residual[high_mask]), -9.0, 9.0))
    if abs(high_gain) >= 1.0 and len(filters) < max_filters:
        eq_filter = EqFilter("high_shelf", 9000.0, 0.75, high_gain)
        filters.append(eq_filter)
        residual -= filter_response_db(eq_filter, GRAPH_FREQS)

    usable = (GRAPH_FREQS >= 35.0) & (GRAPH_FREQS <= 10000.0)
    for _ in range(max_filters - len(filters)):
        masked = np.where(usable, residual, 0.0)
        index = int(np.argmax(np.abs(masked)))
        gain = float(np.clip(masked[index], -8.0, 8.0))
        if abs(gain) < 0.9:
            break
        freq = float(np.clip(GRAPH_FREQS[index], 20.0, 20000.0))
        q = _quality_from_width(GRAPH_FREQS, residual, index)
        eq_filter = EqFilter("peaking", round(freq), q, gain).sanitized()
        filters.append(eq_filter)
        residual -= filter_response_db(eq_filter, GRAPH_FREQS)
        residual = _smooth_log_curve(residual, window=17)

    return Preset(name=_autoeq_name(device_curve, target_curve), filters=filters).sanitized()


def _autoeq_name(device_curve: FrequencyCurve, target_curve: FrequencyCurve) -> str:
    stamp = datetime.now().strftime("%H:%M")
    return f"AutoEQ {device_curve.name} to {target_curve.name} {stamp}"
