from __future__ import annotations

from aieq.models import EqFilter, Preset
from aieq.storage import PresetStore


def test_preset_store_can_find_by_name_case_insensitive(tmp_path) -> None:
    store = PresetStore(tmp_path / "presets.sqlite3")
    saved = store.save_new(Preset("Warm", [EqFilter()]))
    found = store.get_preset_by_name("warm")
    assert found is not None
    assert found.id == saved.id

