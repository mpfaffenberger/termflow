"""Resize detection: read_key timeouts + widget repaint-on-resize."""

from __future__ import annotations

import os
from io import StringIO

from termflow.ansi.utils import visible_length
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder, read_key


class TestReadKeyTimeout:
    def test_timeout_returns_empty_string(self):
        r, w = os.pipe()
        stream = os.fdopen(r, "r")
        try:
            assert read_key(stream, timeout=0.01) == ""
        finally:
            stream.close()
            os.close(w)

    def test_key_arrives_within_timeout(self):
        r, w = os.pipe()
        stream = os.fdopen(r, "r")
        try:
            os.write(w, b"a")
            assert read_key(stream, timeout=1.0) == "a"
        finally:
            stream.close()
            os.close(w)

    def test_eof_reads_as_escape_not_timeout(self):
        # A dead terminal must cancel the menu, never spin the poll loop.
        r, w = os.pipe()
        os.close(w)
        stream = os.fdopen(r, "r")
        try:
            assert read_key(stream, timeout=0.05) == "escape"
        finally:
            stream.close()

    def test_escape_sequences_still_parse_with_timeout(self):
        r, w = os.pipe()
        stream = os.fdopen(r, "r")
        try:
            os.write(w, b"\x1b[A")
            assert read_key(stream, timeout=1.0) == "up"
        finally:
            stream.close()
            os.close(w)


def _frames(raw: str) -> list[str]:
    return raw.split("\x1b[H")[1:]


class TestMenuResize:
    def test_resize_repaints_at_new_width(self):
        sizes = {"wh": (80, 24)}
        calls = {"n": 0}

        def key_source() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                sizes["wh"] = (40, 12)  # simulate a resize mid-wait
                return ""  # timeout tick
            return "escape"

        out = StringIO()
        long_label = "a" * 70
        result = (
            MenuBuilder("Resize me")
            .items([MenuItem(long_label)])
            .key_source(key_source)
            .output(out)
            .size(lambda: sizes["wh"])
            .alt_screen(False)
            .run()
        )
        assert result.cancelled
        frames = _frames(out.getvalue())
        # One initial paint plus one resize repaint -- no keypress needed.
        assert len(frames) == 2
        for line in frames[1].replace("\x1b[J", "").split("\r\n"):
            assert visible_length(line.replace("\x1b[K", "")) <= 40

    def test_timeout_without_resize_does_not_repaint(self):
        script = iter(["", "", "escape"])
        out = StringIO()
        result = (
            MenuBuilder("Static")
            .items([MenuItem("one")])
            .key_source(lambda: next(script))
            .output(out)
            .size(lambda: (60, 20))
            .alt_screen(False)
            .run()
        )
        assert result.cancelled
        assert len(_frames(out.getvalue())) == 1  # idle ticks stay silent


class TestRightEdgePadding:
    """Full-width content must stop one column short of the terminal edge.

    Writing the last column arms xterm's deferred-wrap state, where the
    trailing clear-to-EOL erases the character just written -- the
    'tok' -> 'to' clipping bug. StringIO can't reproduce the terminal
    behavior, so we pin the contract: no painted line ever reaches the
    full reported width.
    """

    def test_menu_lines_stay_one_column_short(self):
        out = StringIO()
        (
            MenuBuilder("Edge")
            .items([MenuItem("x" * 120, description="y" * 120)])
            .key_source(iter(["escape"]).__next__)
            .output(out)
            .size(lambda: (40, 12))
            .alt_screen(False)
            .run()
        )
        for frame in _frames(out.getvalue()):
            for line in frame.replace("\x1b[J", "").split("\r\n"):
                assert visible_length(line.replace("\x1b[K", "")) <= 39

    def test_textinput_lines_stay_one_column_short(self):
        out = StringIO()
        (
            TextInputBuilder("T" * 80)
            .initial("v" * 80)
            .key_source(iter(["end", "escape"]).__next__)
            .output(out)
            .size(lambda: (40, 12))
            .alt_screen(False)
            .run()
        )
        for frame in _frames(out.getvalue()):
            for line in frame.replace("\x1b[J", "").split("\r\n"):
                assert visible_length(line.replace("\x1b[K", "")) <= 39


class TestTextInputResize:
    def test_resize_repaints_at_new_width(self):
        sizes = {"wh": (80, 24)}
        calls = {"n": 0}

        def key_source() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                sizes["wh"] = (30, 8)
                return ""
            return "escape"

        out = StringIO()
        result = (
            TextInputBuilder("A title easily longer than thirty columns")
            .initial("some starting text that overflows")
            .key_source(key_source)
            .output(out)
            .size(lambda: sizes["wh"])
            .alt_screen(False)
            .run()
        )
        assert result.cancelled
        frames = _frames(out.getvalue())
        assert len(frames) == 2
        for line in frames[1].replace("\x1b[J", "").split("\r\n"):
            assert visible_length(line.replace("\x1b[K", "")) <= 30
