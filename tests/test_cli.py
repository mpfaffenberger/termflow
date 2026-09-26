"""CLI: live width, --pager, and file error handling."""

from __future__ import annotations

import pytest

from termflow import cli
from termflow.ansi import visible, visible_length
from termflow.tui.pager import Pager

WORDS = " ".join(f"word{i}" for i in range(60))


@pytest.fixture
def doc(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(f"# Title\n\n{WORDS}\n", encoding="utf-8")
    return path


class TestRenderFile:
    def test_fixed_width_wraps_at_word_boundaries(self, doc, capsys):
        assert cli.main([str(doc), "--width", "30", "--no-clipboard"]) == 0
        out = capsys.readouterr().out
        assert all(visible_length(line) <= 30 for line in out.splitlines())
        assert set(WORDS.split()) <= set(visible(out).split())

    def test_missing_file_is_an_error(self, tmp_path, capsys):
        assert cli.main([str(tmp_path / "nope.md")]) == 1
        assert "File not found" in capsys.readouterr().err

    def test_pager_falls_back_to_plain_output_when_not_a_tty(self, doc, capsys):
        # capsys stdout is not a tty -> behave like `cat`, never open a pager.
        assert cli.main([str(doc), "--pager", "--width", "30"]) == 0
        assert "word59" in visible(capsys.readouterr().out)


class TestPageRender:
    def test_builds_a_reflowing_markdown_pager(self, doc, monkeypatch):
        opened: list[Pager] = []
        monkeypatch.setattr(Pager, "run", lambda self: opened.append(self))
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)

        assert cli.main([str(doc), "--pager"]) == 0

        (pager,) = opened
        assert pager._title == "doc.md"
        # The CLI's config max_width (default 120) caps the reflow width.
        lines = pager._reflow(500)
        assert all(visible_length(line) <= 120 for line in lines)
        assert len(pager._reflow(40)) > len(lines)
