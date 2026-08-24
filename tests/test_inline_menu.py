"""Inline (inquirer-style) menu rendering: paint at the cursor, not the screen."""

from __future__ import annotations

from io import StringIO

from termflow.tui import MenuBuilder, MenuItem


def drive(keys, **overrides):
    script = iter(keys)
    out = StringIO()
    builder = (
        MenuBuilder(overrides.pop("title", "Pick"))
        .items(overrides.pop("items", [MenuItem("one"), MenuItem("two")]))
        .inline()
        .key_source(lambda: next(script))
        .output(out)
        .size(lambda: (60, 20))
    )
    for name, value in overrides.items():
        getattr(builder, name)(value)
    return builder.run(), out.getvalue()


class TestInlineMenu:
    def test_never_touches_alt_screen_or_cursor_home(self):
        _, raw = drive(["escape"])
        assert "\x1b[?1049h" not in raw  # no alt screen
        assert "\x1b[H" not in raw  # no cursor home
        assert "one" in raw

    def test_repaint_climbs_back_to_first_row(self):
        import re

        _, raw = drive(["down", "enter"])
        climbs = re.findall(r"\r\x1b\[(\d+)A", raw)
        assert climbs, "no cursor-up repaint found"
        # The climb must equal the previous paint's line count, so the
        # repaint lands exactly on the menu's first row.
        first_paint = re.split(r"\r\x1b\[\d+A", raw)[0]
        assert int(climbs[0]) == first_paint.count("\r\n")

    def test_selection_still_works(self):
        result, _ = drive(["down", "enter"])
        assert result.item.label == "two"

    def test_inline_overrides_alt_screen_default(self):
        # .inline() must win even though use_alt_screen defaults True.
        _, raw = drive(["escape"])
        assert "\x1b[?1049" not in raw

    def test_preview_renders_inline_too(self):
        _, raw = drive(
            ["escape"],
            items=[MenuItem("a")],
            preview=lambda item: f"PV::{item.label}",
            size=lambda: (100, 20),
        )
        assert "PV::a" in raw
