"""Tests for live mode primitives: cell buffer, diffing, pixels, input."""

from __future__ import annotations

import random
import re

from termflow.live.buffer import (
    BOLD,
    DEFAULT,
    LOWER_HALF,
    WIDE_TAIL,
    Rect,
    ScreenBuffer,
    render_diff,
    rgb,
)
from termflow.live.input import KeyEvent, KeyState, VTInputParser
from termflow.live.pixels import UPPER_HALF, PixelSurface
from termflow.tui.keys import Key

# -- a tiny VT interpreter, just enough to replay render_diff output -------

_SEQ = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")


def replay(screen: list[list[str]], ansi: str) -> None:
    """Apply CUP / ED / SGR(ignored) + text to a char grid, in place."""
    row = col = 0
    pos = 0
    while pos < len(ansi):
        m = _SEQ.match(ansi, pos)
        if m:
            params, final = m.groups()
            if final == "H":
                r, c = [*params.split(";"), "1", "1"][:2]
                row, col = int(r) - 1, int(c) - 1
            elif final == "J":
                for line in screen:
                    line[:] = [" "] * len(line)
            pos = m.end()
            continue
        ch = ansi[pos]
        screen[row][col] = ch
        col += 2 if ch == "界" else 1
        if ch == "界":
            screen[row][col - 1] = WIDE_TAIL
        pos += 1


def visible(buffer: ScreenBuffer) -> list[list[str]]:
    w = buffer.width
    return [buffer.chars[y * w : (y + 1) * w] for y in range(buffer.height)]


# -- buffer / regions ------------------------------------------------------


class TestRegion:
    def test_text_is_clipped_to_region(self):
        buf = ScreenBuffer(10, 3)
        buf.region(Rect(2, 1, 4, 1)).text(0, 0, "abcdefgh")
        assert buf.row_text(1) == "  abcd    "
        assert buf.row_text(0).strip() == ""

    def test_negative_coordinates_clip(self):
        buf = ScreenBuffer(5, 1)
        buf.region().text(-2, 0, "abcdefg")
        assert buf.row_text(0) == "cdefg"

    def test_sub_region_is_clipped_to_parent(self):
        buf = ScreenBuffer(10, 3)
        parent = buf.region(Rect(0, 0, 4, 3))
        parent.sub(Rect(2, 0, 10, 1)).text(0, 0, "XXXXXX")
        assert buf.row_text(0) == "  XX      "

    def test_box_draws_border_and_title(self):
        buf = ScreenBuffer(12, 3)
        buf.region().box("Hi", fg=rgb(1, 2, 3))
        assert buf.row_text(0).startswith("┌─ Hi ─")
        assert buf.row_text(0).endswith("┐")
        assert buf.row_text(1) == "│" + " " * 10 + "│"
        assert buf.row_text(2) == "└" + "─" * 10 + "┘"
        assert buf.attrs[3] & BOLD  # title is bold

    def test_ansi_truecolor_bold_and_reset(self):
        buf = ScreenBuffer(6, 1)
        buf.region().ansi(0, 0, "\x1b[1;38;2;255;0;0ma\x1b[0mb\x1b[48;5;196mc")
        assert buf.row_text(0).startswith("abc")
        assert (buf.fg[0], buf.attrs[0]) == (rgb(255, 0, 0), BOLD)
        assert (buf.fg[1], buf.attrs[1]) == (DEFAULT, 0)
        assert buf.bg[2] == rgb(255, 0, 0)

    def test_ansi_skips_hyperlinks(self):
        buf = ScreenBuffer(8, 1)
        buf.region().ansi(0, 0, "\x1b]8;;https://x.y\x1b\\link\x1b]8;;\x1b\\!")
        assert buf.row_text(0) == "link!   "

    def test_wide_chars_take_two_cells(self):
        buf = ScreenBuffer(4, 1)
        end = buf.region().text(0, 0, "界a")
        assert end == 3
        assert buf.chars[:3] == ["界", WIDE_TAIL, "a"]

    def test_wide_char_cut_by_edge_is_blanked(self):
        buf = ScreenBuffer(3, 1)
        buf.region().text(2, 0, "界")
        assert buf.chars[2] == " "


# -- diff rendering ----------------------------------------------------------


class TestRenderDiff:
    def test_first_frame_is_a_full_repaint(self):
        buf = ScreenBuffer(3, 2)
        buf.region().text(0, 0, "hey")
        out = render_diff(None, buf)
        assert out.startswith("\x1b[0m\x1b[2J")
        assert "hey" in out

    def test_identical_frames_emit_nothing(self):
        a, b = ScreenBuffer(5, 2), ScreenBuffer(5, 2)
        assert render_diff(a, b) == ""

    def test_single_change_moves_once(self):
        a, b = ScreenBuffer(10, 3), ScreenBuffer(10, 3)
        b.region().set(7, 2, "x")
        assert render_diff(a, b) == "\x1b[3;8Hx\x1b[0m"  # frames start in the default style

    def test_size_change_repaints(self):
        assert "\x1b[2J" in render_diff(ScreenBuffer(4, 4), ScreenBuffer(5, 4))

    def test_only_changed_color_component_is_sent(self):
        a, b = ScreenBuffer(2, 1), ScreenBuffer(2, 1)
        region = b.region()
        region.set(0, 0, "▀", rgb(1, 1, 1), rgb(9, 9, 9))
        region.set(1, 0, "▀", rgb(2, 2, 2), rgb(9, 9, 9))
        out = render_diff(a, b)
        assert out.count("48;2;9;9;9") == 1  # bg unchanged: not repeated
        assert "\x1b[38;2;2;2;2m" in out

    def test_round_trip_matches_buffer(self):
        rng = random.Random(7)
        w, h = 16, 6
        screen = [[" "] * w for _ in range(h)]
        prev = None
        for _ in range(40):
            cur = ScreenBuffer(w, h)
            if prev is not None:
                cur.chars, cur.fg = list(prev.chars), list(prev.fg)
                cur.bg, cur.attrs = list(prev.bg), list(prev.attrs)
            region = cur.region()
            for _ in range(rng.randint(0, 8)):
                text = rng.choice(["a", "bc", "界", "xyz", " "])
                region.text(rng.randrange(w), rng.randrange(h), text, fg=rng.randrange(3))
            replay(screen, render_diff(prev, cur))
            assert screen == visible(cur)
            prev = cur


def replay_pixels(screen: list[list[tuple[int, int]]], ansi: str, state: list[int]) -> None:
    """Replay diff output, tracking colors: each cell -> (top, bottom).

    ``state`` is the terminal's [fg, bg] and persists across frames,
    exactly like a real terminal's SGR state does.
    """
    row = col = 0
    pos = 0
    while pos < len(ansi):
        m = _SEQ.match(ansi, pos)
        if m:
            params, final = m.groups()
            if final == "H":
                r, c = params.split(";")
                row, col = int(r) - 1, int(c) - 1
            elif final == "m":
                codes = [int(p) for p in params.split(";")]
                i = 0
                while i < len(codes):
                    if codes[i] in (38, 48):
                        color = rgb(*codes[i + 2 : i + 5])
                        state[0 if codes[i] == 38 else 1] = color
                        i += 5
                        continue
                    if codes[i] in (0, 39):
                        state[0] = DEFAULT
                    if codes[i] in (0, 49):
                        state[1] = DEFAULT
                    i += 1
            pos = m.end()
            continue
        fg, bg = state
        screen[row][col] = {UPPER_HALF: (fg, bg), LOWER_HALF: (bg, fg), " ": (bg, bg)}[ansi[pos]]
        col += 1
        pos += 1


class TestPixelDiffColors:
    def test_swaps_and_solid_cells_reproduce_every_pixel(self):
        rng = random.Random(3)
        palette = [rgb(255, 0, 0), rgb(0, 255, 0), rgb(0, 0, 255)]  # few colors: many swaps
        w, h = 12, 4
        screen = [[(0, 0)] * w for _ in range(h)]
        state = [DEFAULT, DEFAULT]
        surface = PixelSurface(w, h * 2)
        prev = None
        for _ in range(30):
            for _ in range(rng.randint(1, 40)):
                surface.set(rng.randrange(w), rng.randrange(h * 2), rng.choice(palette))
            cur = ScreenBuffer(w, h)
            surface.draw(cur.region())
            out = render_diff(prev, cur)
            replay_pixels(screen, out, state)
            expected = [
                [(cur.fg[y * w + x], cur.bg[y * w + x]) for x in range(w)] for y in range(h)
            ]
            assert screen == expected
            prev = cur
        assert LOWER_HALF in out or " " in out  # the tricks actually fired

    def test_default_colors_are_never_swapped(self):
        buf = ScreenBuffer(2, 1)
        region = buf.region()
        region.set(0, 0, UPPER_HALF, rgb(1, 2, 3), DEFAULT)
        region.set(1, 0, UPPER_HALF, DEFAULT, rgb(1, 2, 3))
        out = render_diff(None, buf)
        assert LOWER_HALF not in out
        assert out.count(UPPER_HALF) == 2


# -- pixels -------------------------------------------------------------------


class TestPixelSurface:
    def test_draw_packs_two_rows_per_cell(self):
        s = PixelSurface(2, 2)
        s.pixels[:] = [1, 2, 3, 4]
        buf = ScreenBuffer(2, 1)
        s.draw(buf.region())
        assert buf.chars == [UPPER_HALF, UPPER_HALF]
        assert buf.fg == [1, 2]
        assert buf.bg == [3, 4]

    def test_odd_height_last_row_gets_black_bottom(self):
        s = PixelSurface(1, 3, color=5)
        buf = ScreenBuffer(1, 2)
        s.draw(buf.region())
        assert (buf.fg[1], buf.bg[1]) == (5, 0)

    def test_fill_rect_clips(self):
        s = PixelSurface(3, 3)
        s.fill_rect(-1, 2, 10, 10, 9)
        assert s.pixels == [0, 0, 0, 0, 0, 0, 9, 9, 9]

    def test_blit_scaled_nearest_neighbor(self):
        s = PixelSurface(4, 2)
        s.blit_scaled([1, 2], 2, 1)
        assert s.pixels == [1, 1, 2, 2, 1, 1, 2, 2]

    def test_resize_reports_change(self):
        s = PixelSurface(2, 2)
        assert not s.resize(2, 2)
        assert s.resize(3, 1)
        assert len(s.pixels) == 3


# -- input -----------------------------------------------------------------------


class TestVTInputParser:
    def test_plain_and_shifted_letters(self):
        assert VTInputParser().feed("wW") == [KeyEvent("w", text="w"), KeyEvent("w", text="W")]

    def test_ctrl_and_named_keys(self):
        keys = [e.key for e in VTInputParser().feed("\x03\r\t\x7f ")]
        assert keys == ["ctrl-c", Key.ENTER, Key.TAB, Key.BACKSPACE, " "]

    def test_arrows_csi_and_ss3(self):
        assert [e.key for e in VTInputParser().feed("\x1b[A\x1bOD\x1b[5~")] == [
            Key.UP,
            Key.LEFT,
            Key.PAGE_UP,
        ]

    def test_sequence_split_across_reads(self):
        p = VTInputParser()
        assert p.feed("\x1b[") == []
        assert p.pending
        assert p.feed("B") == [KeyEvent(Key.DOWN)]

    def test_lone_escape_resolves_on_flush(self):
        p = VTInputParser()
        assert p.feed("\x1b") == []
        assert p.flush() == [KeyEvent(Key.ESCAPE)]

    def test_kitty_query_reply_enables_release_support(self):
        p = VTInputParser()
        assert p.feed("\x1b[?11u") == []
        assert p.kitty

    def test_kitty_press_repeat_release(self):
        events = VTInputParser().feed("\x1b[97u\x1b[97;1:2u\x1b[97;1:3u")
        assert [(e.key, e.pressed, e.repeat) for e in events] == [
            ("a", True, False),
            ("a", True, True),
            ("a", False, False),
        ]

    def test_kitty_modifiers(self):
        events = VTInputParser().feed("\x1b[97;2u\x1b[99;5u\x1b[57441u\x1b[1;1:3A")
        assert events[0] == KeyEvent("a", text="A")
        assert events[1].key == "ctrl-c"
        assert events[2].key == "shift"
        assert events[3] == KeyEvent(Key.UP, pressed=False)


class TestKeyState:
    def test_true_release_holds_until_released(self):
        ks = KeyState()
        ks.feed(KeyEvent("w"), 0.0, reports_release=True)
        ks.expire(100.0)
        assert ks.is_down("w")
        ks.feed(KeyEvent("w", pressed=False), 100.0, reports_release=True)
        assert not ks.is_down("w")

    def test_synthetic_hold_expires_and_repeat_extends(self):
        ks = KeyState(initial_hold=0.5, repeat_hold=0.1)
        ks.feed(KeyEvent("w"), 0.0, reports_release=False)
        ks.expire(0.4)
        assert ks.is_down("w")
        ks.feed(KeyEvent("w"), 0.45, reports_release=False)  # autorepeat
        ks.expire(0.54)
        assert ks.is_down("w")
        ks.expire(0.56)
        assert not ks.is_down("w")

    def test_was_pressed_only_on_transition(self):
        ks = KeyState()
        ks.feed(KeyEvent(" "), 0.0, True)
        assert ks.was_pressed(" ")
        ks.end_frame()
        ks.feed(KeyEvent(" ", repeat=True), 0.1, True)
        assert not ks.was_pressed(" ")

    def test_ctrl_combo_release_frees_base_key(self):
        ks = KeyState()
        ks.feed(KeyEvent("ctrl-c"), 0.0, True)
        assert ks.is_down("c")
        ks.feed(KeyEvent("c", pressed=False), 0.1, True)
        assert not ks.is_down("c")

    def test_axis(self):
        ks = KeyState()
        ks.feed(KeyEvent(Key.RIGHT), 0.0, True)
        assert ks.axis(("a", Key.LEFT), ("d", Key.RIGHT)) == 1
        ks.release_all()
        assert ks.axis(("a",), ("d",)) == 0
