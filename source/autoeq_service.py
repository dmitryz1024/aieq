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


def build_autoeq_preset_result(
    device_curve: FrequencyCurve,
    target_curve: FrequencyCurve,
    *,
    max_filters: int = 12,
    backend: str | None = None,
) -> AutoEqPresetResult:
    backend = (backend or os.environ.get("AIEQ_AUTOEQ_BACKEND", "auto")).strip().lower() or "auto"
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


def build_autoeq_preset(
    device_curve: FrequencyCurve,
    target_curve: FrequencyCurve,
    *,
    max_filters: int = 12,
    backend: str | None = None,
) -> Preset:
    return build_autoeq_preset_result(device_curve, target_curve, max_filters=max_filters, backend=backend).preset


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
    measurement.process(
        target=target,
        min_mean_error=True,
        fs=int(DEFAULT_SAMPLE_RATE),
        concha_interference=False,
    )
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


def _build_local_autoeq_preset(
    device_curve: FrequencyCurve,
    target_curve: FrequencyCurve,
    *,
    max_filters: int = 12,
) -> Preset:
    device = device_curve.response_db(GRAPH_FREQS)
    target = target_curve.response_db(GRAPH_FREQS)
    desired = np.clip(target - device, -18.0, 18.0)
    residual = _smooth_log_curve(desired, window=15)
    weights = _fit_weights(GRAPH_FREQS)
    filters: list[EqFilter] = []

    for _ in range(max_filters):
        candidate = _best_local_filter(GRAPH_FREQS, residual, weights)
        if candidate is None or abs(candidate.gain) < 0.35:
            break
        filters.append(candidate)
        residual -= filter_response_db(candidate, GRAPH_FREQS)
        residual = _smooth_log_curve(residual, window=9)
        if _weighted_error(residual, weights) < 0.10:
            break

    filters = _prune_tiny_filters(filters)
    filters.sort(key=lambda item: (item.freq, item.type))
    return Preset(name=_autoeq_name(device_curve, target_curve), filters=filters).sanitized()


def _fit_weights(freqs: np.ndarray) -> np.ndarray:
    weights = np.ones_like(freqs, dtype=np.float64)
    weights[(freqs >= 80.0) & (freqs <= 9000.0)] = 1.35
    weights[(freqs < 35.0) | (freqs > 16000.0)] = 0.45
    return weights


def _weighted_error(residual: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean(weights * np.square(residual)))


def _best_local_filter(freqs: np.ndarray, residual: np.ndarray, weights: np.ndarray) -> EqFilter | None:
    base_error = _weighted_error(residual, weights)
    candidates: list[tuple[str, float, float]] = []
    for index in _residual_peak_indexes(freqs, residual, limit=22):
        freq = float(np.clip(freqs[index], 20.0, 20000.0))
        for q in (0.45, 0.6, 0.8, 1.0, 1.35, 1.8, 2.5, 3.5, 5.0, 7.0):
            candidates.append(("peaking", freq, q))
    for freq in (55.0, 80.0, 105.0, 140.0, 190.0):
        candidates.append(("low_shelf", freq, 0.7))
    for freq in (5000.0, 6500.0, 8000.0, 10000.0, 12500.0):
        candidates.append(("high_shelf", freq, 0.7))

    best_filter: EqFilter | None = None
    best_error = base_error
    for filter_type, freq, q in candidates:
        unit = filter_response_db(EqFilter(filter_type, freq, q, 1.0), freqs)
        denominator = float(np.sum(weights * unit * unit))
        if denominator <= 1e-9:
            continue
        gain = float(np.sum(weights * residual * unit) / denominator)
        gain = float(np.clip(gain, -9.0, 9.0))
        if abs(gain) < 0.25:
            continue
        eq_filter = EqFilter(filter_type, round(freq), q, gain).sanitized()
        next_residual = residual - filter_response_db(eq_filter, freqs)
        error = _weighted_error(next_residual, weights)
        if error < best_error - 0.004:
            best_error = error
            best_filter = eq_filter
    return best_filter


def _residual_peak_indexes(freqs: np.ndarray, residual: np.ndarray, *, limit: int) -> list[int]:
    usable = (freqs >= 30.0) & (freqs <= 14500.0)
    order = np.argsort(np.abs(np.where(usable, residual, 0.0)))[::-1]
    selected: list[int] = []
    min_log_distance = 0.035
    for raw_index in order:
        index = int(raw_index)
        if not usable[index] or abs(residual[index]) < 0.25:
            break
        log_freq = float(np.log10(freqs[index]))
        if any(abs(log_freq - float(np.log10(freqs[item]))) < min_log_distance for item in selected):
            continue
        selected.append(index)
        if len(selected) >= limit:
            break
    return selected


def _prune_tiny_filters(filters: list[EqFilter]) -> list[EqFilter]:
    return [item for item in filters if abs(item.gain) >= 0.25]


def _autoeq_name(device_curve: FrequencyCurve, target_curve: FrequencyCurve) -> str:
    stamp = datetime.now().strftime("%H:%M")
    return f"AutoEQ {device_curve.name} to {target_curve.name} {stamp}"
