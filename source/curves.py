from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


CURVES_DIR = app_root() / "curves"
DEVICE_CURVES_DIR = CURVES_DIR / "devices"
TARGET_CURVES_DIR = CURVES_DIR / "targets"
SUPPORTED_CURVE_SUFFIXES = {".csv", ".txt"}


@dataclass(frozen=True, slots=True)
class FrequencyCurve:
    name: str
    freqs: np.ndarray
    db: np.ndarray
    path: Path | None = None
    root: Path | None = None

    def normalized(self, reference_hz: float = 1000.0) -> "FrequencyCurve":
        if self.is_lazy:
            return self.loaded().normalized(reference_hz)
        if self.freqs.size == 0:
            return self
        reference = float(np.interp(reference_hz, self.freqs, self.db))
        return FrequencyCurve(self.name, self.freqs, self.db - reference, self.path, self.root)

    def response_db(self, freqs: np.ndarray) -> np.ndarray:
        if self.is_lazy:
            return self.loaded().response_db(freqs)
        if self.freqs.size == 0:
            return np.zeros_like(freqs, dtype=np.float64)
        return np.interp(freqs, self.freqs, self.db, left=self.db[0], right=self.db[-1])

    @property
    def is_lazy(self) -> bool:
        return self.path is not None and self.freqs.size == 0 and self.db.size == 0

    def loaded(self) -> "FrequencyCurve":
        if not self.is_lazy:
            return self
        if self.path is None:
            return self
        return load_curve_file(self.path, root=self.root)


def ensure_curve_dirs() -> None:
    DEVICE_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_CURVES_DIR.mkdir(parents=True, exist_ok=True)


def default_device_curve() -> FrequencyCurve:
    return FrequencyCurve("Default", np.array([20.0, 20000.0]), np.array([0.0, 0.0]))


def load_curve_file(path: Path, *, root: Path | None = None) -> FrequencyCurve:
    freqs: list[float] = []
    values: list[float] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = _split_curve_line(line)
        if len(parts) < 2:
            continue
        try:
            freq = float(parts[0].replace(",", "."))
            value = float(parts[1].replace(",", "."))
        except ValueError:
            continue
        if freq > 0 and np.isfinite(freq) and np.isfinite(value):
            freqs.append(freq)
            values.append(value)
    if not freqs:
        raise ValueError(f"Curve file has no frequency data: {path}")
    order = np.argsort(freqs)
    sorted_freqs = np.asarray(freqs, dtype=np.float64)[order]
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    return FrequencyCurve(_curve_name_from_path(path, root=root), sorted_freqs, sorted_values, path, root).normalized()


def _split_curve_line(line: str) -> list[str]:
    if ";" in line:
        return [part.strip() for part in line.split(";")]
    if "," in line:
        return [part.strip() for part in line.split(",")]
    return line.split()


def _curve_name_from_path(path: Path, *, root: Path | None = None) -> str:
    if root is None:
        return path.stem
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.stem
    if len(relative.parts) == 1:
        return path.stem

    parents = relative.with_suffix("").parts[:-1]
    meaningful_parts = [part for part in parents if part.casefold() not in {"data", "raw", "processed"}]
    if not meaningful_parts:
        return path.stem
    context = " / ".join(meaningful_parts[-2:])
    return f"{path.stem} [{context}]"


def _iter_curve_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in SUPPORTED_CURVE_SUFFIXES
            and not any(part.startswith(".") for part in path.relative_to(folder).parts)
        ),
        key=lambda item: (item.stem.casefold(), str(item.parent).casefold()),
    )


def list_curves(folder: Path, *, include_default: bool = False, lazy: bool = False) -> list[FrequencyCurve]:
    ensure_curve_dirs()
    curves = [default_device_curve()] if include_default else []
    for path in _iter_curve_files(folder):
        if lazy:
            curves.append(FrequencyCurve(_curve_name_from_path(path, root=folder), np.array([]), np.array([]), path, folder))
            continue
        try:
            curves.append(load_curve_file(path, root=folder))
        except ValueError:
            continue
    return curves
