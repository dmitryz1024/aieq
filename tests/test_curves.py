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
