from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from .dsp import DEFAULT_SAMPLE_RATE, StreamingFir, design_fir_from_preset
from .models import Preset, flat_preset

SAMPLE_RATE_OPTIONS: tuple[int, ...] = (44100, 48000, 88200, 96000, 176400, 192000, 384000)
AUDIO_DTYPE_LABELS: dict[str, str] = {
    "float32": "32-bit float",
    "int32": "32-bit PCM",
    "int16": "16-bit PCM",
}
DEFAULT_AUDIO_BLOCK_SIZE = 0
DEFAULT_AUDIO_LATENCY: str | float = 0.05
OUTPUT_LIMITER_CEILING = 0.98


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_latency(name: str, default: str | float) -> str | float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    lowered = value.casefold()
    if lowered in {"low", "high"}:
        return lowered
    try:
        if lowered.endswith("ms"):
            return max(0.0, float(lowered[:-2].strip()) / 1000.0)
        return max(0.0, float(lowered))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class AudioDevice:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def label(self) -> str:
        return f"[{self.hostapi}] {self.name}"


@dataclass(frozen=True, slots=True)
class AudioStreamSetting:
    sample_rate: int
    dtype: str

    @property
    def dtype_label(self) -> str:
        return AUDIO_DTYPE_LABELS.get(self.dtype, self.dtype)


def _sounddevice() -> Any:
    import sounddevice as sd

    return sd


def refresh_audio_backend() -> None:
    sd = _sounddevice()
    terminate = getattr(sd, "_terminate", None)
    initialize = getattr(sd, "_initialize", None)
    if not callable(terminate) or not callable(initialize):
        return
    terminate()
    initialize()


def list_audio_devices(kind: str) -> list[AudioDevice]:
    sd = _sounddevice()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    result: list[AudioDevice] = []
    for index, raw in enumerate(devices):
        max_inputs = int(raw.get("max_input_channels", 0))
        max_outputs = int(raw.get("max_output_channels", 0))
        if kind == "input" and max_inputs <= 0:
            continue
        if kind == "output" and max_outputs <= 0:
            continue
        hostapi_name = hostapis[int(raw["hostapi"])]["name"]
        result.append(
            AudioDevice(
                index=index,
                name=str(raw["name"]),
                hostapi=str(hostapi_name),
                max_input_channels=max_inputs,
                max_output_channels=max_outputs,
                default_samplerate=float(raw.get("default_samplerate") or DEFAULT_SAMPLE_RATE),
            )
        )
    return result


def list_supported_stream_settings(input_device: AudioDevice, output_device: AudioDevice) -> list[AudioStreamSetting]:
    sd = _sounddevice()
    channels = max(1, min(2, input_device.max_input_channels, output_device.max_output_channels))
    default_rates = [round(input_device.default_samplerate), round(output_device.default_samplerate)]
    sample_rates = sorted({rate for rate in [*SAMPLE_RATE_OPTIONS, *default_rates] if rate > 0})
    settings: list[AudioStreamSetting] = []
    for sample_rate in sample_rates:
        for dtype in AUDIO_DTYPE_LABELS:
            try:
                sd.check_input_settings(
                    device=input_device.index,
                    channels=channels,
                    dtype=dtype,
                    samplerate=float(sample_rate),
                )
                sd.check_output_settings(
                    device=output_device.index,
                    channels=channels,
                    dtype=dtype,
                    samplerate=float(sample_rate),
                )
            except Exception:  # noqa: BLE001 - unsupported combinations are expected here.
                continue
            settings.append(AudioStreamSetting(sample_rate=int(sample_rate), dtype=dtype))
    if settings:
        return settings
    fallback_rate = int(output_device.default_samplerate or input_device.default_samplerate or DEFAULT_SAMPLE_RATE)
    return [AudioStreamSetting(sample_rate=fallback_rate, dtype="float32")]


class AudioEngine:
    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        block_size: int | None = None,
        fir_taps: int = 1025,
        crossfade_frames: int = 4096,
        latency: str | float | None = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.dtype = "float32"
        self.block_size = int(block_size if block_size is not None else _env_int("AIEQ_AUDIO_BLOCK_SIZE", DEFAULT_AUDIO_BLOCK_SIZE))
        self.fir_taps = int(fir_taps)
        self.crossfade_frames = int(crossfade_frames)
        self.latency = latency if latency is not None else _env_latency("AIEQ_AUDIO_LATENCY", DEFAULT_AUDIO_LATENCY)
        self.stream: Any | None = None
        self.channels = 2
        self.processor = StreamingFir(design_fir_from_preset(flat_preset(), self.sample_rate, self.fir_taps), channels=2)
        self.fade_from_processor: StreamingFir | None = None
        self.crossfade_remaining = 0
        self.lock = Lock()
        self.last_error: str | None = None
        self.last_status: str | None = None

    @property
    def is_running(self) -> bool:
        return self.stream is not None

    @property
    def output_latency_ms(self) -> float | None:
        stream = self.stream
        if stream is None:
            return None
        latency = getattr(stream, "latency", None)
        if isinstance(latency, (tuple, list)):
            if len(latency) < 2:
                return None
            latency = latency[1]
        if latency is None:
            return None
        try:
            return float(latency) * 1000.0
        except (TypeError, ValueError):
            return None

    def update_preset(self, preset: Preset) -> None:
        taps = design_fir_from_preset(preset.sanitized(self.sample_rate), self.sample_rate, self.fir_taps)
        next_processor = StreamingFir(taps, channels=self.channels)
        with self.lock:
            if self.crossfade_frames > 0:
                self.fade_from_processor = self.processor
                self.crossfade_remaining = self.crossfade_frames
            else:
                self.fade_from_processor = None
                self.crossfade_remaining = 0
            self.processor = next_processor

    def start(
        self,
        input_device: AudioDevice,
        output_device: AudioDevice,
        preset: Preset,
        *,
        sample_rate: float | None = None,
        dtype: str = "float32",
    ) -> None:
        if self.stream is not None:
            return

        sd = _sounddevice()
        self.sample_rate = float(sample_rate or output_device.default_samplerate or input_device.default_samplerate or self.sample_rate)
        self.dtype = dtype
        self.channels = max(1, min(2, input_device.max_input_channels, output_device.max_output_channels))
        self.processor = StreamingFir(
            design_fir_from_preset(preset.sanitized(self.sample_rate), self.sample_rate, self.fir_taps),
            channels=self.channels,
        )
        self.fade_from_processor = None
        self.crossfade_remaining = 0
        self.last_error = None
        self.last_status = None
        uses_mme = self._uses_mme(input_device, output_device)
        latency = self._effective_latency(input_device, output_device)
        stream_kwargs = {
            "device": (input_device.index, output_device.index),
            "samplerate": self.sample_rate,
            "blocksize": self.block_size,
            "dtype": self.dtype,
            "channels": self.channels,
            "latency": (latency, latency),
            "callback": self._callback,
        }
        if not uses_mme:
            stream_kwargs.update(
                {
                    "clip_off": True,
                    "dither_off": True,
                    "prime_output_buffers_using_stream_callback": True,
                }
            )
        stream = self._open_stream(sd, stream_kwargs)
        self.stream = stream
        stream.start()

    def _open_stream(self, sd: Any, stream_kwargs: dict[str, Any]) -> Any:
        attempts: list[dict[str, Any]] = [dict(stream_kwargs)]
        for latency in (("low", "low"), ("high", "high")):
            if latency != stream_kwargs.get("latency"):
                fallback = dict(stream_kwargs)
                fallback["latency"] = latency
                attempts.append(fallback)
        fallback = dict(stream_kwargs)
        fallback.pop("latency", None)
        attempts.append(fallback)
        conservative = dict(fallback)
        conservative.pop("clip_off", None)
        conservative.pop("dither_off", None)
        conservative.pop("prime_output_buffers_using_stream_callback", None)
        attempts.append(conservative)

        last_error: Exception | None = None
        for attempt in attempts:
            try:
                return sd.Stream(**attempt)
            except Exception as exc:  # noqa: BLE001 - trying progressively safer PortAudio options.
                last_error = exc
        if last_error is not None:
            raise last_error
        return sd.Stream(**stream_kwargs)

    def _effective_latency(self, input_device: AudioDevice, output_device: AudioDevice) -> str | float:
        if self._uses_mme(input_device, output_device):
            return "high"
        return self.latency

    @staticmethod
    def _uses_mme(input_device: AudioDevice, output_device: AudioDevice) -> bool:
        return "mme" in input_device.hostapi.casefold() or "mme" in output_device.hostapi.casefold()

    def stop(self) -> None:
        stream = self.stream
        self.stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        if status:
            self.last_status = str(status)
        try:
            input_float = self._stream_to_float(indata)
            with self.lock:
                out = self._process_locked(input_float)
            out = self._limit_output(out)
            np.copyto(outdata, self._float_to_stream(out, outdata.dtype))
        except Exception as exc:  # noqa: BLE001 - audio callbacks must never raise.
            self.last_error = str(exc)
            np.copyto(outdata, indata if indata.shape == outdata.shape else np.zeros_like(outdata))

    def _process_locked(self, indata: np.ndarray) -> np.ndarray:
        fade_from = self.fade_from_processor
        if fade_from is None or self.crossfade_remaining <= 0:
            self.fade_from_processor = None
            self.crossfade_remaining = 0
            return self.processor.process(indata)

        old = fade_from.process(indata)
        new = self.processor.process(indata)
        frames = new.shape[0]
        done = self.crossfade_frames - self.crossfade_remaining
        alpha = (np.arange(frames, dtype=np.float32) + done + 1.0) / max(1, self.crossfade_frames)
        alpha = np.clip(alpha, 0.0, 1.0)[:, None]
        out = old * (1.0 - alpha) + new * alpha
        self.crossfade_remaining -= frames
        if self.crossfade_remaining <= 0:
            self.fade_from_processor = None
            self.crossfade_remaining = 0
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _stream_to_float(data: np.ndarray) -> np.ndarray:
        array = np.asarray(data)
        if np.issubdtype(array.dtype, np.floating):
            return array.astype(np.float32, copy=False)
        if np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)
            scale = float(max(abs(info.min), info.max))
            return array.astype(np.float32) / scale
        return array.astype(np.float32)

    @staticmethod
    def _float_to_stream(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
        target_dtype = np.dtype(dtype)
        if np.issubdtype(target_dtype, np.floating):
            return data.astype(target_dtype, copy=False)
        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(target_dtype)
            scaled = np.clip(data, -1.0, 1.0) * float(info.max)
            return scaled.astype(target_dtype)
        return data.astype(target_dtype)

    @staticmethod
    def _limit_output(data: np.ndarray, ceiling: float = OUTPUT_LIMITER_CEILING) -> np.ndarray:
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak <= ceiling or peak <= 0.0:
            return data
        return (data * (ceiling / peak)).astype(np.float32, copy=False)
