"""Headless tests for the Pager widget: scripted keys, StringIO, no tty."""

from __future__ import annotations

from io import StringIO

from termflow.ansi.utils import visible, visible_length
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


class TestReflow:
    """Content given as reflow(width) re-renders whenever the width changes."""

    DOC = "# Title\n\n" + " ".join(f"word{i}" for i in range(200))

    def _resizing(self, sizes, keys):
        """Key source that applies each queued size as a timeout tick first."""
        pending = list(keys)

        def key_source():
            if sizes["queue"]:
                sizes["wh"] = sizes["queue"].pop(0)
                return ""  # timeout tick -> resize repaint
            return pending.pop(0)

        return key_source

    def test_markdown_rewraps_on_resize(self):
        sizes = {"wh": (80, 20), "queue": [(40, 20)]}
        out = StringIO()
        pager = (
            PagerBuilder("Doc")
            .markdown(self.DOC)
            .key_source(self._resizing(sizes, ["q"]))
            .output(out)
            .size(lambda: sizes["wh"])
            .alt_screen(False)
            .build()
        )
        wide_count = pager.line_count
        pager.run()
        assert pager.line_count > wide_count  # narrower -> more lines
        frames = out.getvalue().split("\x1b[H")[1:]
        assert len(frames) == 2
        body = frames[1].replace("\x1b[J", "").replace("\x1b[K", "")
        for line in body.split("\r\n"):
            assert visible_length(line) <= 39
        # Words are never split: any fragment of "wordNN" would carry a
        # digit without being a real document word ("d12", "12", ...).
        doc_words = set(self.DOC.split())
        numbered = [w for w in visible(body).split() if any(c.isdigit() for c in w)]
        assert numbered and all(w in doc_words for w in numbered)

    def test_reflow_only_reruns_when_width_changes(self):
        calls = []

        def reflow(width):
            calls.append(width)
            return [f"{width}"] * 5

        sizes = {"wh": (60, 20), "queue": [(60, 10), (50, 10)]}
        pager = Pager(
            "t",
            reflow=reflow,
            key_source=self._resizing(sizes, ["j", "q"]),
            output=StringIO(),
            size=lambda: sizes["wh"],
            use_alt_screen=False,
        )
        pager.run()
        assert calls == [59, 49]  # height-only change didn't re-render

    def test_resize_keeps_relative_position(self):
        sizes = {"wh": (80, 14)}
        pager = Pager(
            "t",
            reflow=lambda width: [f"line {i}" for i in range(100 if width > 50 else 200)],
            key_source=lambda: "q",
            output=StringIO(),
            size=lambda: sizes["wh"],
            use_alt_screen=False,
        )
        pager.scroll(50)
        assert pager.top == 50
        sizes["wh"] = (40, 14)
        assert pager.line_count == 200
        assert pager.top == 100  # same spot, half-way through

    def test_markdown_respects_max_width(self):
        pager = (
            PagerBuilder("Doc")
            .markdown(self.DOC, max_width=30)
            .size(lambda: (120, 20))
            .output(StringIO())
            .build()
        )
        assert pager.line_count > 1
        assert all(visible_length(ln) <= 30 for ln in pager._content())
