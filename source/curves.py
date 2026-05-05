from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CURVES_DIR = Path("curves")
DEVICE_CURVES_DIR = CURVES_DIR / "devices"
TARGET_CURVES_DIR = CURVES_DIR / "targets"


@dataclass(frozen=True, slots=True)
class FrequencyCurve:
    name: str
    freqs: np.ndarray
    db: np.ndarray
    path: Path | None = None

    def normalized(self, reference_hz: float = 1000.0) -> "FrequencyCurve":
        if self.freqs.size == 0:
            return self
        reference = float(np.interp(reference_hz, self.freqs, self.db))
        return FrequencyCurve(self.name, self.freqs, self.db - reference, self.path)

    def response_db(self, freqs: np.ndarray) -> np.ndarray:
        if self.freqs.size == 0:
            return np.zeros_like(freqs, dtype=np.float64)
        return np.interp(freqs, self.freqs, self.db, left=self.db[0], right=self.db[-1])


def ensure_curve_dirs() -> None:
    DEVICE_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_CURVES_DIR.mkdir(parents=True, exist_ok=True)


def default_device_curve() -> FrequencyCurve:
    return FrequencyCurve("Default", np.array([20.0, 20000.0]), np.array([0.0, 0.0]))


def load_curve_file(path: Path) -> FrequencyCurve:
    freqs: list[float] = []
    values: list[float] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", ".").replace(";", " ").split()
        if len(parts) < 2:
            continue
        try:
            freq = float(parts[0])
            value = float(parts[1])
        except ValueError:
            continue
        if freq > 0:
            freqs.append(freq)
            values.append(value)
    if not freqs:
        raise ValueError(f"Curve file has no frequency data: {path}")
    order = np.argsort(freqs)
    sorted_freqs = np.asarray(freqs, dtype=np.float64)[order]
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    return FrequencyCurve(path.stem, sorted_freqs, sorted_values, path).normalized()


def list_curves(folder: Path, *, include_default: bool = False) -> list[FrequencyCurve]:
    ensure_curve_dirs()
    curves = [default_device_curve()] if include_default else []
    for path in sorted(folder.glob("*.txt"), key=lambda item: item.stem.lower()):
        try:
            curves.append(load_curve_file(path))
        except ValueError:
            continue
    return curves

