from __future__ import annotations

import numpy as np

from source.dsp import GRAPH_FREQS, design_fir_from_preset, envelope_response_db, filter_response_db
from source.models import EqFilter, Preset
from source.audio import AudioEngine
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
