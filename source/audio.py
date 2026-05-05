from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from .dsp import DEFAULT_SAMPLE_RATE, StreamingFir, design_fir_from_preset
from .models import Preset, flat_preset


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


class AudioEngine:
    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        block_size: int = 512,
        fir_taps: int = 1025,
        crossfade_frames: int = 4096,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.block_size = int(block_size)
        self.fir_taps = int(fir_taps)
        self.crossfade_frames = int(crossfade_frames)
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

    def start(self, input_device: AudioDevice, output_device: AudioDevice, preset: Preset) -> None:
        if self.stream is not None:
            return

        sd = _sounddevice()
        self.sample_rate = float(output_device.default_samplerate or input_device.default_samplerate or self.sample_rate)
        self.channels = max(1, min(2, input_device.max_input_channels, output_device.max_output_channels))
        self.processor = StreamingFir(
            design_fir_from_preset(preset.sanitized(self.sample_rate), self.sample_rate, self.fir_taps),
            channels=self.channels,
        )
        self.fade_from_processor = None
        self.crossfade_remaining = 0
        self.last_error = None
        self.last_status = None
        self.stream = sd.Stream(
            device=(input_device.index, output_device.index),
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            channels=self.channels,
            callback=self._callback,
        )
        self.stream.start()

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
            with self.lock:
                out = self._process_locked(indata)
            np.copyto(outdata, out)
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
