"""Headless tests for the Pager widget: scripted keys, StringIO, no tty."""

from __future__ import annotations

from io import StringIO

from termflow.ansi.utils import visible_length
from termflow.tui import PagerBuilder
from termflow.tui.pager import Pager, PagerResult

CONTENT = [f"line {i}" for i in range(50)]


def build(keys, *, lines=None, height=14, width=60, **kwargs):
    script = iter(keys)
    out = StringIO()
    builder = (
        PagerBuilder("Test Pager")
        .lines(lines if lines is not None else CONTENT)
        .key_source(lambda: next(script))
        .output(out)
        .size(lambda: (width, height))
        .alt_screen(False)
    )
    for name, value in kwargs.items():
        if name == "on_key":
            builder.on_key(*value)
        else:
            getattr(builder, name)(value)
    return builder.build(), out


def run(keys, **kwargs):
    pager, out = build(keys, **kwargs)
    result = pager.run()
    return pager, result, out.getvalue()


class TestScrolling:
    def test_j_k_move_one_line(self):
        pager, _, _ = run(["j", "j", "j", "k", "q"])
        assert pager.top == 2

    def test_arrows_match_vim_keys(self):
        pager, _, _ = run(["down", "down", "up", "q"])
        assert pager.top == 1

    def test_scroll_clamps_at_top(self):
        pager, _, _ = run(["k", "k", "q"])
        assert pager.top == 0

    def test_page_and_half_page(self):
        # viewport = height 14 - 4 chrome = 10
        pager, _, _ = run(["f", "u", "q"])
        assert pager.top == 5
        pager, _, _ = run([" ", "b", "q"])
        assert pager.top == 0
        pager, _, _ = run(["page-down", "page-down", "page-up", "q"])
        assert pager.top == 10

    def test_g_G_jump_to_edges(self):
        pager, _, _ = run(["G", "q"])
        assert pager.top == 40  # 50 lines - 10 viewport
        pager, _, _ = run(["G", "g", "q"])
        assert pager.top == 0

    def test_home_end_keys(self):
        pager, _, _ = run(["end", "home", "q"])
        assert pager.top == 0

    def test_scroll_beyond_bottom_clamps(self):
        pager, _, _ = run(["G", "j", "j", "q"])
        assert pager.top == 40

    def test_short_content_never_scrolls(self):
        pager, _, _ = run(["j", "f", "G", "q"], lines=["a", "b"])
        assert pager.top == 0


class TestClosing:
    def test_q_closes_uncancelled(self):
        _, result, _ = run(["q"])
        assert result.key == "q" and not result.cancelled

    def test_enter_closes_uncancelled(self):
        _, result, _ = run(["enter"])
        assert not result.cancelled

    def test_escape_and_ctrl_c_cancel(self):
        _, result, _ = run(["escape"])
        assert result.cancelled
        _, result, _ = run(["ctrl-c"])
        assert result.cancelled


class TestPainting:
    def test_shows_title_and_visible_window(self):
        _, _, out = run(["G", "q"])
        assert "Test Pager" in out
        assert "line 49" in out  # bottom visible after G
        first_frame = out.split("\x1b[H")[1]
        assert "line 0" in first_frame and "line 20" not in first_frame

    def test_position_indicator(self):
        _, _, out = run(["q"])
        assert "Top" in out
        _, _, out = run(["G", "q"])
        assert "Bot" in out
        _, _, out = run(["f", "q"])
        assert "%" in out
        _, _, out = run(["q"], lines=["a"])
        assert "All" in out

    def test_every_line_fits_width(self):
        long_lines = ["x" * 200] * 30
        _, _, out = run(["j", "f", "q"], lines=long_lines, width=40)
        for frame in out.split("\x1b[H"):
            for line in frame.split("\r\n"):
                assert visible_length(line.replace("\x1b[K", "").replace("\x1b[J", "")) <= 39

    def test_resize_repaints(self):
        # Model a stateful terminal: the key source shrinks the window,
        # then returns the poll-timeout sentinel so the loop notices.
        terminal = {"size": (60, 14)}
        script = iter(["resize", "q"])
        out = StringIO()

        def keys():
            key = next(script)
            if key == "resize":
                terminal["size"] = (40, 10)
                return ""  # poll tick
            return key

        pager = Pager(
            "T",
            lines=CONTENT,
            key_source=keys,
            output=out,
            size=lambda: terminal["size"],
            use_alt_screen=False,
        )
        pager.run()
        assert out.getvalue().count("\x1b[H") == 2

    def test_text_convenience_splits_lines(self):
        pager = Pager(
            "T",
            text="a\nb\nc",
            output=StringIO(),
            use_alt_screen=False,
            key_source=iter(["q"]).__next__,
            size=lambda: (40, 10),
        )
        assert pager.line_count == 3


class TestOnKey:
    def test_handler_ends_run_with_key(self):
        _, result, _ = run(["x"], on_key=("x", lambda _pager: PagerResult(key="x")))
        assert result.key == "x"

    def test_handler_returning_none_continues(self):
        seen = []

        def spy(pager):
            seen.append(pager.top)
            return None

        _, result, _ = run(["j", "x", "q"], on_key=("x", spy))
        assert seen == [1]
        assert result.key == "q"

    def test_handler_exception_is_swallowed(self):
        def boom(_pager):
            raise RuntimeError("nope")

        _, result, _ = run(["x", "q"], on_key=("x", boom))
        assert result.key == "q"


def test_builder_on_key_signature():
    # PagerBuilder.on_key takes (key, handler) -- the test helper above
    # adapts a tuple; verify the real fluent form too.
    script = iter(["z"])
    result = (
        PagerBuilder("T")
        .lines(["a"])
        .on_key("z", lambda _pager: PagerResult(key="z"))
        .key_source(lambda: next(script))
        .output(StringIO())
        .size(lambda: (40, 10))
        .alt_screen(False)
        .run()
    )
    assert result.key == "z"
