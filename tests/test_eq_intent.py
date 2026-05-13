from __future__ import annotations

from source.eq_intent import try_build_intent_preset
from source.models import EqFilter, Preset, flat_preset


def test_exact_boost_request_adds_filter_to_mentioned_preset() -> None:
    saved = [
        Preset("Warm", [EqFilter("peaking", 1000, 1, 1)]),
        Preset("AIEQ 2026-05-06 | 01-59-52", [EqFilter("high_shelf", 9000, 0.7, 2)]),
    ]
    result = try_build_intent_preset(
        'Хочу доработать пресет "AIEQ 2026-05-06 | 01-59-52" добавив широкий подъем на 3дб вверх с центром в 2000Гц',
        flat_preset(),
        saved,
    )
    assert result is not None
    assert len(result.preset.filters) == 2
    added = result.preset.filters[-1]
    assert added.type == "peaking"
    assert added.freq == 2000
    assert added.gain == 3
    assert added.q == 0.7


def test_exact_cut_request_uses_negative_gain() -> None:
    result = try_build_intent_preset("убери 2.5 дб в районе 300 гц", flat_preset(), [])
    assert result is not None
    added = result.preset.filters[-1]
    assert added.freq == 300
    assert added.gain == -2.5


def test_english_exact_boost_request() -> None:
    result = try_build_intent_preset("add a wide +3 db boost centered at 2000 hz", flat_preset(), [])
    assert result is not None
    added = result.preset.filters[-1]
    assert added.freq == 2000
    assert added.gain == 3
    assert added.q == 0.7


def test_general_cool_sound_uses_template() -> None:
    result = try_build_intent_preset("сделай короче крутой звук", flat_preset(), [])
    assert result is not None
    assert len(result.preset.filters) >= 3
    assert "энергич" in result.assistant_message.casefold()
