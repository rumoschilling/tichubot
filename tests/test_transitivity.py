"""transitivity: cached ladder x panel evaluation and the report."""

import os

import torch

from tests.test_crn_eval import _save
from training import transitivity


def test_ladder_panel_report_and_cache(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)  # keep the repo's passing fallback out
    try:
        a, b, x = _save(tmp_path / "a.pt", 1), _save(tmp_path / "b.pt", 2), _save(tmp_path / "x.pt", 3)
        out = tmp_path / "out"
        out.mkdir()
        ladder, panel = [("a", a), ("b", b)], [("x", x)]
        cells = {label: transitivity.evaluate((label, path), panel, out, deals=8, procs=1) for label, path in ladder}
        first = sorted(out.iterdir())
        again = transitivity.evaluate(("a", a), panel, out, deals=8, procs=1)
        assert again == cells["a"] and sorted(out.iterdir()) == first  # cached, no new file
        pair = transitivity.evaluate(("a", a), [("b", b)], out, deals=8, procs=1)
        text = transitivity.report(ladder, panel, cells, [("a", "b", pair["b"])])
        assert "x" in text and "direct" in text and "via x" in text
    finally:
        os.chdir(cwd)
