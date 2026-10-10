"""Tests for terminal color-mode detection and the 256-color fallback.

Terminals without 24-bit support (macOS Terminal.app is the common one) don't
ignore ``ESC[38;2;R;G;Bm``: they read each number as its own SGR code, so a
component such as 106 or 104 turns into a bright cyan or blue *background*.
"""

import re
from io import StringIO

import pytest

from termflow import render_markdown
from termflow.ansi import (
    bg_color,
    detect_color_mode,
    fg_color,
    get_color_mode,
    rgb_to_ansi256,
    set_color_mode,
)
from termflow.syntax import Highlighter

TRUECOLOR_SGR = re.compile(r"\x1b\[(?:[0-9]+;)*[34]8;2;")


class TestDetectColorMode:
    """detect_color_mode mirrors Rich: COLORTERM decides, Windows is truecolor."""

    @pytest.mark.parametrize("value", ["truecolor", "24bit", " TrueColor "])
    def test_colorterm_truecolor(self, value):
        assert detect_color_mode({"COLORTERM": value}, platform="darwin") == "truecolor"

    def test_no_colorterm_is_256(self):
        environ = {"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"}
        assert detect_color_mode(environ, platform="darwin") == "256"

    def test_other_colorterm_is_256(self):
        assert detect_color_mode({"COLORTERM": "yes"}, platform="linux") == "256"

    def test_windows_is_truecolor(self):
        assert detect_color_mode({}, platform="win32") == "truecolor"


class TestColorModeState:
    def test_set_and_get(self):
        set_color_mode("256")
        assert get_color_mode() == "256"
        set_color_mode("truecolor")
        assert get_color_mode() == "truecolor"

    def test_none_restores_detection(self, monkeypatch):
        monkeypatch.setenv("COLORTERM", "truecolor")
        set_color_mode("256")
        set_color_mode(None)
        assert get_color_mode() == "truecolor"

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            set_color_mode("16")  # type: ignore[arg-type]


class TestRgbToAnsi256:
    @pytest.mark.parametrize(
        ("rgb", "index"),
        [
            ((0, 0, 0), 16),
            ((255, 255, 255), 231),
            ((255, 0, 0), 196),
            ((95, 135, 175), 67),  # exact 6x6x6 cube entry
            ((128, 128, 128), 244),  # exact greyscale-ramp entry
            ((8, 8, 8), 232),  # darkest grey
        ],
    )
    def test_known_colors(self, rgb, index):
        assert rgb_to_ansi256(*rgb) == index

    def test_always_in_extended_range(self):
        for value in range(0, 256, 15):
            assert 16 <= rgb_to_ansi256(value, 255 - value, value // 2) <= 255


class TestColorGeneration256:
    def test_fg_color(self):
        set_color_mode("256")
        assert fg_color("#FF0000") == "\x1b[38;5;196m"

    def test_bg_color(self):
        set_color_mode("256")
        assert bg_color("#000000") == "\x1b[48;5;16m"

    def test_invalid_hex_still_empty(self):
        set_color_mode("256")
        assert fg_color("nope") == ""
        assert bg_color("nope") == ""

    def test_truecolor_unchanged(self):
        set_color_mode("truecolor")
        assert fg_color("#9ece6a") == "\x1b[38;2;158;206;106m"


class TestHighlighterColorMode:
    def test_default_follows_mode(self):
        set_color_mode("256")
        assert Highlighter().true_color is False
        set_color_mode("truecolor")
        assert Highlighter().true_color is True

    def test_explicit_true_color_wins(self):
        set_color_mode("256")
        assert Highlighter(true_color=True).true_color is True


class TestNoTruecolorIn256Mode:
    """Regression: a fenced shell block on Terminal.app showed cyan/blue bars."""

    MARKDOWN = (
        "Run this:\n\n"
        "```sh\n"
        'echo "COLORTERM=$COLORTERM TERM=$TERM"\n'
        "uname -a\n"
        "```\n\n"
        "See [the docs](https://example.com) for more.\n"
    )

    def test_render_markdown_emits_no_truecolor(self):
        set_color_mode("256")
        output = StringIO()
        render_markdown(self.MARKDOWN, output=output, width=80)
        rendered = output.getvalue()
        assert "\x1b[38;5;" in rendered
        assert not TRUECOLOR_SGR.search(rendered)

    def test_render_markdown_truecolor_when_supported(self):
        set_color_mode("truecolor")
        output = StringIO()
        render_markdown(self.MARKDOWN, output=output, width=80)
        assert TRUECOLOR_SGR.search(output.getvalue())
