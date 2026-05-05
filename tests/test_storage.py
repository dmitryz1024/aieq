from __future__ import annotations

import gc
import time
from pathlib import Path
from uuid import uuid4

from source.models import EqFilter, Preset
from source.storage import PresetStore


def test_preset_store_can_find_by_name_case_insensitive() -> None:
    path = Path(f"_test_presets_{uuid4().hex}.sqlite3")
    try:
        store = PresetStore(path)
        saved = store.save_new(Preset("Warm", [EqFilter()]))
        found = store.get_preset_by_name("warm")
        assert found is not None
        assert found.id == saved.id
    finally:
        gc.collect()
        for _ in range(10):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.05)
