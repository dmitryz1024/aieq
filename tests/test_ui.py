from __future__ import annotations

from source.models import EqFilter
from source.ui import LEGEND_LABEL_MAX_CHARS, MainWindow, elide_middle


def test_elide_middle_keeps_short_text() -> None:
    assert elide_middle("AIEQ") == "AIEQ"


def test_elide_middle_truncates_long_text_from_the_middle() -> None:
    text = "AutoEQ dmitryz1024 | 2026-05-06 | 01-59-52 – Fiio JH5 to Harman 2019"
    shortened = elide_middle(text)
    assert len(shortened) == LEGEND_LABEL_MAX_CHARS
    assert shortened.startswith("AutoEQ")
    assert shortened.endswith("Harman 2019")
    assert "..." in shortened


def test_preset_signature_matches_editor_visible_precision() -> None:
    stored = [EqFilter(type="peaking", freq=1000.6, q=1.23456, gain=1.236, enabled=True)]
    editor_visible = [EqFilter(type="peaking", freq=1001.0, q=1.235, gain=1.24, enabled=True)]

    assert MainWindow.filter_signature(stored) == MainWindow.filter_signature(editor_visible)
