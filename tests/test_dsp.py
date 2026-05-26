from __future__ import annotations

import numpy as np

from source.dsp import GRAPH_FREQS, design_fir_from_preset, envelope_response_db, filter_response_db, preset_response_db
from source.models import EqFilter, Preset
from source.audio import AudioDevice, AudioEngine, list_supported_stream_settings
from source.curves import FrequencyCurve
from source.autoeq_service import build_autoeq_preset, build_autoeq_preset_result


def test_zero_gain_peaking_is_flat() -> None:
    response = filter_response_db(EqFilter("peaking", 1000.0, 1.0, 0.0), GRAPH_FREQS)
    assert np.max(np.abs(response)) < 1e-9


def test_envelope_mixing_does_not_sum_identical_boosts() -> None:
    filters = [
        EqFilter("peaking", 1000.0, 1.0, 6.0),
        EqFilter("peaking", 1000.0, 1.0, 6.0),
    ]
    freqs = np.array([1000.0])
    response = envelope_response_db(filters, freqs)
    assert 5.8 < response[0] < 6.2


def test_fir_design_is_finite_and_odd_length() -> None:
    preset = Preset("Test", [EqFilter("low_shelf", 100.0, 0.7, 3.0)])
    taps = design_fir_from_preset(preset, num_taps=513)
    assert taps.shape == (513,)
    assert np.isfinite(taps).all()


def test_audio_engine_crossfades_updated_preset() -> None:
    engine = AudioEngine(crossfade_frames=16, fir_taps=129)
    preset = Preset("Boost", [EqFilter("high_shelf", 8000.0, 0.7, 3.0)])
    engine.update_preset(preset)
    block = np.zeros((8, 2), dtype=np.float32)
    out = engine._process_locked(block)
    assert out.shape == block.shape
    assert engine.fade_from_processor is not None


def test_audio_engine_converts_integer_stream_samples() -> None:
    samples = np.array([[-32768, 32767]], dtype=np.int16)
    as_float = AudioEngine._stream_to_float(samples)
    assert as_float.dtype == np.float32
    assert as_float[0, 0] <= -0.99
    assert as_float[0, 1] >= 0.99

    restored = AudioEngine._float_to_stream(np.array([[-1.0, 1.0]], dtype=np.float32), np.dtype("int16"))
    assert restored.dtype == np.int16
    assert restored[0, 0] < 0
    assert restored[0, 1] > 0


def test_audio_engine_requests_target_latency_stream(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeStream:
        latency = (0.01, 0.02)

        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def start(self) -> None:
            pass

    class FakeSd:
        Stream = FakeStream

    monkeypatch.delenv("AIEQ_AUDIO_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("AIEQ_AUDIO_LATENCY", raising=False)
    monkeypatch.setattr("source.audio._sounddevice", lambda: FakeSd())
    input_device = AudioDevice(1, "Input", "WASAPI", 2, 0, 48000)
    output_device = AudioDevice(2, "Output", "WASAPI", 0, 2, 48000)
    engine = AudioEngine(fir_taps=129)
    engine.start(input_device, output_device, Preset("Flat", []), sample_rate=48000)

    assert captured["blocksize"] == 0
    assert captured["latency"] == ("low", "low")
    assert captured["clip_off"] is True
    assert captured["dither_off"] is True
    assert engine.output_latency_ms == 20.0


def test_audio_engine_falls_back_to_low_latency_if_target_latency_fails(monkeypatch) -> None:
    attempts: list[object] = []

    class FakeStream:
        latency = (0.01, 0.02)

        def __init__(self, **kwargs) -> None:
            attempts.append(kwargs["latency"])
            if kwargs["latency"] != ("low", "low"):
                raise ValueError("unsupported latency")

        def start(self) -> None:
            pass

    class FakeSd:
        Stream = FakeStream

    monkeypatch.delenv("AIEQ_AUDIO_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("AIEQ_AUDIO_LATENCY", raising=False)
    monkeypatch.setattr("source.audio._sounddevice", lambda: FakeSd())
    input_device = AudioDevice(1, "Input", "WASAPI", 2, 0, 48000)
    output_device = AudioDevice(2, "Output", "WASAPI", 0, 2, 48000)
    engine = AudioEngine(latency=0.05, fir_taps=129)
    engine.start(input_device, output_device, Preset("Flat", []), sample_rate=48000)

    assert attempts == [(0.05, 0.05), ("low", "low")]


def test_audio_engine_uses_low_latency_without_extra_flags_for_mme(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeStream:
        latency = (0.08, 0.08)

        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def start(self) -> None:
            pass

    class FakeSd:
        Stream = FakeStream

    monkeypatch.delenv("AIEQ_AUDIO_LATENCY", raising=False)
    monkeypatch.setattr("source.audio._sounddevice", lambda: FakeSd())
    input_device = AudioDevice(1, "Input", "MME", 2, 0, 48000)
    output_device = AudioDevice(2, "Output", "MME", 0, 2, 48000)
    engine = AudioEngine(fir_taps=129)
    engine.start(input_device, output_device, Preset("Flat", []), sample_rate=48000)

    assert captured["latency"] == ("low", "low")
    assert "clip_off" not in captured
    assert "dither_off" not in captured
    assert "prime_output_buffers_using_stream_callback" not in captured


def test_audio_engine_allows_explicit_custom_latency_for_mme(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeStream:
        latency = (0.03, 0.03)

        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def start(self) -> None:
            pass

    class FakeSd:
        Stream = FakeStream

    monkeypatch.setattr("source.audio._sounddevice", lambda: FakeSd())
    input_device = AudioDevice(1, "Input", "MME", 2, 0, 48000)
    output_device = AudioDevice(2, "Output", "MME", 0, 2, 48000)
    engine = AudioEngine(fir_taps=129)
    engine.set_latency(("low", 0.03), custom=True)
    engine.start(input_device, output_device, Preset("Flat", []), sample_rate=48000)

    assert captured["latency"] == ("low", 0.03)


def test_audio_limiter_prevents_overflow_without_latency() -> None:
    hot = np.array([[0.5, -2.0], [1.4, 0.2]], dtype=np.float32)
    limited = AudioEngine._limit_output(hot)
    assert np.max(np.abs(limited)) <= 0.98 + 1e-6
    assert limited.shape == hot.shape


def test_audio_settings_are_checked_per_input_and_output(monkeypatch) -> None:
    class FakeSd:
        def check_input_settings(self, *, samplerate, **kwargs):
            if int(samplerate) not in {48000, 192000}:
                raise ValueError("unsupported input")

        def check_output_settings(self, *, samplerate, **kwargs):
            if int(samplerate) != 192000:
                raise ValueError("unsupported output")

    monkeypatch.setattr("source.audio._sounddevice", lambda: FakeSd())
    input_device = AudioDevice(1, "Input", "WASAPI", 2, 0, 192000)
    output_device = AudioDevice(2, "Output", "WASAPI", 0, 2, 192000)
    settings = list_supported_stream_settings(input_device, output_device)
    assert {setting.sample_rate for setting in settings} == {192000}


def test_autoeq_service_builds_filters_from_curves(monkeypatch) -> None:
    monkeypatch.setenv("AIEQ_AUTOEQ_BACKEND", "local")
    freqs = np.geomspace(20, 20000, 128)
    device = FrequencyCurve("Device", freqs, np.sin(np.linspace(0, 4, 128)) * 3)
    target = FrequencyCurve("Target", freqs, np.zeros_like(freqs))
    preset = build_autoeq_preset(device, target, max_filters=5)
    assert preset.name.startswith("AutoEQ")
    assert 1 <= len(preset.filters) <= 5


def test_autoeq_local_backend_can_be_forced(monkeypatch) -> None:
    monkeypatch.setenv("AIEQ_AUTOEQ_BACKEND", "local")
    freqs = np.geomspace(20, 20000, 128)
    device = FrequencyCurve("Device", freqs, np.sin(np.linspace(0, 4, 128)) * 3)
    target = FrequencyCurve("Target", freqs, np.zeros_like(freqs))
    result = build_autoeq_preset_result(device, target, max_filters=5)
    assert result.backend == "local"
    assert result.preset.name.startswith("AutoEQ")


def test_autoeq_backend_argument_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("AIEQ_AUTOEQ_BACKEND", "official")
    freqs = np.geomspace(20, 20000, 128)
    device = FrequencyCurve("Device", freqs, np.sin(np.linspace(0, 4, 128)) * 3)
    target = FrequencyCurve("Target", freqs, np.zeros_like(freqs))
    result = build_autoeq_preset_result(device, target, max_filters=5, backend="local")
    assert result.backend == "local"


def test_autoeq_local_backend_uses_precise_fitter() -> None:
    freqs = np.geomspace(20, 20000, 192)
    device = FrequencyCurve(
        "Device",
        freqs,
        (
            np.exp(-np.square(np.log(freqs / 110.0)) / 0.40) * 4.0
            - np.exp(-np.square(np.log(freqs / 1800.0)) / 0.18) * 3.0
            + np.exp(-np.square(np.log(freqs / 5200.0)) / 0.10) * 4.5
        ),
    )
    target = FrequencyCurve("Target", freqs, np.zeros_like(freqs))

    result = build_autoeq_preset_result(device, target, max_filters=8, backend="local")
    desired = target.response_db(GRAPH_FREQS) - device.response_db(GRAPH_FREQS)
    error = np.mean(np.abs(desired - preset_response_db(result.preset, GRAPH_FREQS)))

    assert result.backend == "local"
    assert len(result.preset.filters) >= 1
    assert error < 0.35
