from __future__ import annotations

import os
import sys
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resolve_app_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    candidates = [
        Path.cwd() / expanded,
        app_root() / expanded,
        Path(__file__).resolve().parent.parent / expanded,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return app_root() / expanded


def load_env_file(path: Path | None = None, *, override: bool = False) -> None:
    env_path = path or resolve_app_path(Path(".env"))
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
