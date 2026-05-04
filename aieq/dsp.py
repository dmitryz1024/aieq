from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import EqFilter, Preset

DEFAULT_SAMPLE_RATE = 48000.0
GRAPH_FREQS = np.geomspace(20.0, 20000.0, 768)


@dataclass(frozen=True, slots=True)
class Biquad:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    def response(self, freqs: np.ndarray, sample_rate: float) -> np.ndarray:
        omega = 2.0 * np.pi * freqs / sample_rate
        z1 = np.exp(-1j * omega)
        z2 = np.exp(-2j * omega)
        return (self.b0 + self.b1 * z1 + self.b2 * z2) / (1.0 + self.a1 * z1 + self.a2 * z2)


def _normalize(b0: float, b1: float, b2: float, a0: float, a1: float, a2: float) -> Biquad:
    if abs(a0) < 1e-12:
        a0 = 1.0
    return Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def biquad_for_filter(eq_filter: EqFilter, sample_rate: float = DEFAULT_SAMPLE_RATE) -> Biquad:
    item = eq_filter.sanitized(sample_rate)
    freq = min(item.freq, sample_rate * 0.475)
    omega = 2.0 * math.pi * freq / sample_rate
    sin_w = math.sin(omega)
    cos_w = math.cos(omega)
    q = max(item.q, 0.1)
    alpha = sin_w / (2.0 * q)
    gain = item.gain
    amp = 10.0 ** (gain / 40.0)

    if item.type == "peaking":
        b0 = 1.0 + alpha * amp
        b1 = -2.0 * cos_w
        b2 = 1.0 - alpha * amp
        a0 = 1.0 + alpha / amp
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha / amp
        return _normalize(b0, b1, b2, a0, a1, a2)

    if item.type in {"low_shelf", "high_shelf"}:
        shelf_slope = max(q, 0.1)
        alpha_shelf = sin_w / 2.0 * math.sqrt(max((amp + 1.0 / amp) * (1.0 / shelf_slope - 1.0) + 2.0, 1e-9))
        sqrt_amp = math.sqrt(amp)

        if item.type == "low_shelf":
            b0 = amp * ((amp + 1.0) - (amp - 1.0) * cos_w + 2.0 * sqrt_amp * alpha_shelf)
            b1 = 2.0 * amp * ((amp - 1.0) - (amp + 1.0) * cos_w)
            b2 = amp * ((amp + 1.0) - (amp - 1.0) * cos_w - 2.0 * sqrt_amp * alpha_shelf)
            a0 = (amp + 1.0) + (amp - 1.0) * cos_w + 2.0 * sqrt_amp * alpha_shelf
            a1 = -2.0 * ((amp - 1.0) + (amp + 1.0) * cos_w)
            a2 = (amp + 1.0) + (amp - 1.0) * cos_w - 2.0 * sqrt_amp * alpha_shelf
            return _normalize(b0, b1, b2, a0, a1, a2)

        b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w + 2.0 * sqrt_amp * alpha_shelf)
        b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w)
        b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w - 2.0 * sqrt_amp * alpha_shelf)
        a0 = (amp + 1.0) - (amp - 1.0) * cos_w + 2.0 * sqrt_amp * alpha_shelf
        a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w)
        a2 = (amp + 1.0) - (amp - 1.0) * cos_w - 2.0 * sqrt_amp * alpha_shelf
        return _normalize(b0, b1, b2, a0, a1, a2)

    if item.type == "low_pass":
        b0 = (1.0 - cos_w) / 2.0
        b1 = 1.0 - cos_w
        b2 = (1.0 - cos_w) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha
        return _normalize(b0, b1, b2, a0, a1, a2)

    if item.type == "high_pass":
        b0 = (1.0 + cos_w) / 2.0
        b1 = -(1.0 + cos_w)
        b2 = (1.0 + cos_w) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha
        return _normalize(b0, b1, b2, a0, a1, a2)

    if item.type == "band_pass":
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha
        return _normalize(b0, b1, b2, a0, a1, a2)

    if item.type == "notch":
        b0 = 1.0
        b1 = -2.0 * cos_w
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha
        return _normalize(b0, b1, b2, a0, a1, a2)

    return _normalize(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def filter_response_db(eq_filter: EqFilter, freqs: np.ndarray = GRAPH_FREQS, sample_rate: float = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    if not eq_filter.enabled:
        return np.zeros_like(freqs, dtype=np.float64)
    biquad = biquad_for_filter(eq_filter, sample_rate)
    magnitude = np.abs(biquad.response(freqs, sample_rate))
    return 20.0 * np.log10(np.maximum(magnitude, 1e-8))


def envelope_response_db(
    filters: list[EqFilter] | tuple[EqFilter, ...],
    freqs: np.ndarray = GRAPH_FREQS,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    result = np.zeros_like(freqs, dtype=np.float64)
    for eq_filter in filters:
        if not eq_filter.enabled:
            continue
        response = filter_response_db(eq_filter, freqs, sample_rate)
        mask = np.abs(response) > np.abs(result)
        result[mask] = response[mask]
    return result


def preset_response_db(preset: Preset, freqs: np.ndarray = GRAPH_FREQS, sample_rate: float = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    return envelope_response_db(preset.filters, freqs, sample_rate)


def design_fir_from_preset(
    preset: Preset,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    num_taps: int = 1025,
    design_fft_size: int = 16384,
) -> np.ndarray:
    if num_taps % 2 == 0:
        num_taps += 1
    fft_size = max(design_fft_size, 2 ** math.ceil(math.log2(num_taps * 8)))
    freqs = np.linspace(0.0, sample_rate / 2.0, fft_size // 2 + 1)
    safe_freqs = np.maximum(freqs, 20.0)
    db = np.clip(preset_response_db(preset, safe_freqs, sample_rate), -60.0, 24.0)
    magnitude = 10.0 ** (db / 20.0)

    impulse_zero_phase = np.fft.irfft(magnitude, fft_size)
    half = num_taps // 2
    taps = np.concatenate((impulse_zero_phase[-half:], impulse_zero_phase[: half + 1]))
    taps *= np.hanning(num_taps)
    return taps.astype(np.float32)


class StreamingFir:
    def __init__(self, taps: np.ndarray, channels: int) -> None:
        self.channels = max(1, int(channels))
        self.set_taps(taps)

    def set_taps(self, taps: np.ndarray) -> None:
        clean = np.asarray(taps, dtype=np.float32).reshape(-1)
        if clean.size < 1:
            clean = np.array([1.0], dtype=np.float32)
        self.taps = clean
        self.history = np.zeros((self.channels, max(0, clean.size - 1)), dtype=np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        data = np.asarray(block, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[1] != self.channels:
            self.channels = data.shape[1]
            self.history = np.zeros((self.channels, max(0, self.taps.size - 1)), dtype=np.float32)

        output = np.empty_like(data)
        for channel in range(self.channels):
            extended = np.concatenate((self.history[channel], data[:, channel]))
            output[:, channel] = np.convolve(extended, self.taps, mode="valid").astype(np.float32)
            if self.taps.size > 1:
                self.history[channel] = extended[-(self.taps.size - 1) :]
        return output

