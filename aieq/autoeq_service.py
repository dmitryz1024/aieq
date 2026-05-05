from __future__ import annotations

from datetime import datetime

import numpy as np

from .curves import FrequencyCurve
from .dsp import GRAPH_FREQS, filter_response_db
from .models import EqFilter, Preset


def _smooth_log_curve(values: np.ndarray, window: int = 31) -> np.ndarray:
    if values.size < window:
        return values.copy()
    kernel = np.hanning(window)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _quality_from_width(freqs: np.ndarray, residual: np.ndarray, index: int) -> float:
    peak = residual[index]
    threshold = abs(peak) * 0.45
    left = index
    right = index
    while left > 0 and abs(residual[left]) >= threshold and np.sign(residual[left]) == np.sign(peak):
        left -= 1
    while right < residual.size - 1 and abs(residual[right]) >= threshold and np.sign(residual[right]) == np.sign(peak):
        right += 1
    bandwidth = max(freqs[right] - freqs[left], freqs[index] * 0.12)
    return float(np.clip(freqs[index] / bandwidth, 0.45, 8.0))


def build_autoeq_preset(device_curve: FrequencyCurve, target_curve: FrequencyCurve, *, max_filters: int = 8) -> Preset:
    device = device_curve.response_db(GRAPH_FREQS)
    target = target_curve.response_db(GRAPH_FREQS)
    residual = _smooth_log_curve(target - device)
    filters: list[EqFilter] = []

    low_mask = (GRAPH_FREQS >= 30.0) & (GRAPH_FREQS <= 130.0)
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

    usable = (GRAPH_FREQS >= 35.0) & (GRAPH_FREQS <= 16000.0)
    for _ in range(max_filters - len(filters)):
        masked = np.where(usable, residual, 0.0)
        index = int(np.argmax(np.abs(masked)))
        gain = float(np.clip(masked[index], -9.0, 9.0))
        if abs(gain) < 0.9:
            break
        freq = float(np.clip(GRAPH_FREQS[index], 20.0, 20000.0))
        q = _quality_from_width(GRAPH_FREQS, residual, index)
        eq_filter = EqFilter("peaking", round(freq), q, gain).sanitized()
        filters.append(eq_filter)
        residual -= filter_response_db(eq_filter, GRAPH_FREQS)

    stamp = datetime.now().strftime("%H:%M")
    name = f"AutoEQ {device_curve.name} to {target_curve.name} {stamp}"
    return Preset(name=name, filters=filters).sanitized()

