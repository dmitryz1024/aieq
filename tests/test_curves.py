from __future__ import annotations

from pathlib import Path

import numpy as np

from source.curves import load_curve_file


def test_curve_file_is_normalized_at_1khz() -> None:
    path = Path("curve.txt")
    try:
        path.write_text("20 100\n1000 110\n20000 90\n", encoding="utf-8")
        curve = load_curve_file(path)
    finally:
        path.unlink(missing_ok=True)
    assert curve.name == "curve"
    assert np.interp(1000.0, curve.freqs, curve.db) == 0.0


def test_curve_file_supports_csv_comma_separator() -> None:
    path = Path("curve_csv.txt")
    try:
        path.write_text("20,81.5\n1000,91.5\n20000,71.5\n", encoding="utf-8")
        curve = load_curve_file(path)
    finally:
        path.unlink(missing_ok=True)
    assert curve.freqs.tolist() == [20.0, 1000.0, 20000.0]
    assert np.interp(1000.0, curve.freqs, curve.db) == 0.0
