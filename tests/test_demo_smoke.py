"""The offline demo runs end to end and prints its result table."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

DEMO = Path(__file__).parent.parent / "examples" / "demo.py"


def test_offline_demo_prints_tables(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(DEMO)])
    runpy.run_path(str(DEMO), run_name="__main__")
    out = capsys.readouterr().out
    assert "Rolling-start results" in out
    assert "Break-even loan rate" in out
    assert "forced disposal" in out
