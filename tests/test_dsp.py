from __future__ import annotations

import numpy as np

from aieq.dsp import GRAPH_FREQS, design_fir_from_preset, envelope_response_db, filter_response_db
from aieq.models import EqFilter, Preset
from aieq.audio import AudioEngine


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
