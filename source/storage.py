from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .models import Preset, utc_now_iso


def default_db_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return base / "AIEQ" / "presets.sqlite3"


class PresetStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_presets_updated ON presets(updated_at DESC)")

    def list_presets(self) -> list[Preset]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, data FROM presets ORDER BY updated_at DESC, id DESC").fetchall()
        presets: list[Preset] = []
        for row in rows:
            try:
                preset = Preset.from_dict(json.loads(row["data"]))
                preset.id = int(row["id"])
                presets.append(preset)
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        return presets

    def get_preset(self, preset_id: int) -> Preset | None:
        with self._connect() as connection:
            row = connection.execute("SELECT id, data FROM presets WHERE id = ?", (preset_id,)).fetchone()
        if row is None:
            return None
        preset = Preset.from_dict(json.loads(row["data"]))
        preset.id = int(row["id"])
        return preset

    def get_preset_by_name(self, name: str) -> Preset | None:
        normalized = name.strip().casefold()
        if not normalized:
            return None
        for preset in self.list_presets():
            if preset.name.casefold() == normalized:
                return preset
        return None

    def save_new(self, preset: Preset, name: str | None = None) -> Preset:
        now = utc_now_iso()
        saved = preset.clone(name=name or preset.name, keep_id=False).sanitized()
        saved.created_at = now
        saved.updated_at = now
        data = json.dumps(saved.to_dict(), ensure_ascii=False, indent=2)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO presets(name, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (saved.name, data, saved.created_at, saved.updated_at),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an id for the saved preset.")
            saved.id = int(cursor.lastrowid)
        return saved

    def update(self, preset: Preset) -> Preset:
        if preset.id is None:
            return self.save_new(preset)
        now = utc_now_iso()
        saved = preset.sanitized()
        saved.updated_at = now
        data = json.dumps(saved.to_dict(), ensure_ascii=False, indent=2)
        with self._connect() as connection:
            connection.execute(
                "UPDATE presets SET name = ?, data = ?, updated_at = ? WHERE id = ?",
                (saved.name, data, saved.updated_at, saved.id),
            )
        return saved

    def delete(self, preset_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
