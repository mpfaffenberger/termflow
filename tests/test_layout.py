"""Responsive split-pane layout: pure helpers + Menu preview collapse."""

from __future__ import annotations

from io import StringIO
from typing import ClassVar

from termflow.ansi.utils import visible_length
from termflow.tui import (
    COLLAPSE_BELOW,
    MenuBuilder,
    MenuItem,
    collapsed,
    split_frame,
    two_columns,
)

DIVIDER = "\u2502"


class TestCollapsed:
    def test_threshold_boundary(self):
        assert collapsed(COLLAPSE_BELOW - 1)
        assert not collapsed(COLLAPSE_BELOW)

    def test_custom_threshold(self):
        assert collapsed(99, threshold=100)
        assert not collapsed(100, threshold=100)


class TestSplitFrame:
    LEFT: ClassVar[list[str]] = ["alpha", "beta"]
    RIGHT: ClassVar[list[str]] = ["one", "two", "three"]

    def test_wide_shows_both_panes_with_divider(self):
        frame = split_frame(self.LEFT, self.RIGHT, width=100, list_width=20, focus="left")
        assert len(frame) == 3  # max(len(left), len(right))
        assert all(DIVIDER in line for line in frame)
        assert "alpha" in frame[0] and "one" in frame[0]

    def test_narrow_left_focus_shows_master_only(self):
        frame = split_frame(self.LEFT, self.RIGHT, width=40, list_width=20, focus="left")
        assert frame == ["alpha", "beta"]
        assert not any(DIVIDER in line for line in frame)

    def test_narrow_right_focus_shows_detail_only(self):
        frame = split_frame(self.LEFT, self.RIGHT, width=40, list_width=20, focus="right")
        assert frame == ["one", "two", "three"]

    def test_narrow_pane_lines_are_truncated_to_width(self):
        frame = split_frame(["x" * 90], [], width=40, list_width=20, focus="left")
        assert visible_length(frame[0]) <= 40

    def test_two_columns_pads_and_truncates(self):
        merged = two_columns(["l"], ["r" * 50], 10, 30)
        assert len(merged) == 1
        assert visible_length(merged[0]) <= 30
        assert merged[0].startswith("l")


class TestMenuPreviewCollapse:
    def drive(self, width):
        out = StringIO()
        (
            MenuBuilder("Pick")
            .items([MenuItem("item-one"), MenuItem("item-two")])
            .preview(lambda item: f"PREVIEW::{item.label}")
            .list_width(24)
            .key_source(iter(["escape"]).__next__)
            .output(out)
            .size(lambda: (width, 20))
            .alt_screen(False)
            .run()
        )
        return out.getvalue()

    def test_wide_terminal_shows_preview(self):
        raw = self.drive(120)
        assert "PREVIEW::item-one" in raw
        assert DIVIDER in raw

    def test_narrow_terminal_hides_preview_list_goes_full_width(self):
        raw = self.drive(60)
        assert "PREVIEW::" not in raw
        assert DIVIDER not in raw
        assert "item-one" in raw

    def test_resize_across_threshold_toggles_preview(self):
        sizes = {"wh": (120, 20)}
        calls = {"n": 0}

        def key_source() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                sizes["wh"] = (60, 20)  # shrink below the threshold
                return ""
            return "escape"

        out = StringIO()
        (
            MenuBuilder("Pick")
            .items([MenuItem("item-one")])
            .preview(lambda item: f"PREVIEW::{item.label}")
            .key_source(key_source)
            .output(out)
            .size(lambda: sizes["wh"])
            .alt_screen(False)
            .run()
        )
        frames = out.getvalue().split("\x1b[H")[1:]
        assert len(frames) == 2
        assert "PREVIEW::" in frames[0]
        assert "PREVIEW::" not in frames[1]
