from __future__ import annotations

from pathlib import Path

import numpy as np

from source.curves import list_curves, load_curve_file


def test_curve_file_is_normalized_at_1khz(tmp_path: Path) -> None:
    path = tmp_path / "curve.txt"
    path.write_text("20 100\n1000 110\n20000 90\n", encoding="utf-8")
    curve = load_curve_file(path)
    assert curve.name == "curve"
    assert np.interp(1000.0, curve.freqs, curve.db) == 0.0


def test_curve_file_supports_csv_comma_separator(tmp_path: Path) -> None:
    path = tmp_path / "curve_csv.csv"
    path.write_text("frequency, raw\n20, 81.5\n1000, 91.5\n20000, 71.5\n", encoding="utf-8")
    curve = load_curve_file(path)
    assert curve.freqs.tolist() == [20.0, 1000.0, 20000.0]
    assert np.interp(1000.0, curve.freqs, curve.db) == 0.0


def test_list_curves_finds_nested_official_csv_files(tmp_path: Path) -> None:
    curve_dir = tmp_path / "devices"
    nested = curve_dir / "autoeq" / "source-name" / "data" / "inear"
    nested.mkdir(parents=True)
    (nested / "Headphone.csv").write_text("frequency,raw\n20,80\n1000,90\n20000,70\n", encoding="utf-8")
    (curve_dir / "Flat.txt").write_text("20 0\n1000 0\n20000 0\n", encoding="utf-8")

    curves = list_curves(curve_dir)

    assert [curve.name for curve in curves] == [
        "Flat",
        "Headphone [source-name / inear]",
    ]


def test_list_curves_can_return_lazy_curve_references(tmp_path: Path) -> None:
    curve_dir = tmp_path / "devices"
    curve_dir.mkdir()
    (curve_dir / "Headphone.csv").write_text("frequency,raw\n20,80\n1000,90\n20000,70\n", encoding="utf-8")

    [curve] = list_curves(curve_dir, lazy=True)

    assert curve.name == "Headphone"
    assert curve.is_lazy
    assert curve.freqs.size == 0
    assert curve.loaded().freqs.tolist() == [20.0, 1000.0, 20000.0]
