"""A scrollable read-only text viewer for terminal UIs.

Build pagers with :class:`PagerBuilder`::

    from termflow.tui import PagerBuilder

    (
        PagerBuilder("Help")
        .lines(["one", "two", "three"])
        .footer_hint("j/k scroll - q close")
        .run()
    )

Everything is injectable (key source, output stream, terminal size) so
pagers are fully testable without a tty -- the same contract as
:class:`termflow.tui.menu.Menu` and :class:`termflow.tui.textinput.TextInput`.

Navigation: arrows / ``j`` / ``k`` scroll by line, PageUp / PageDown /
``b`` / ``f`` / Space by viewport, ``d`` / ``u`` by half viewport,
``g`` / Home and ``G`` / End jump to the edges. ``q``, Enter, Esc and
Ctrl+C close. Content lines may contain ANSI styling; every painted
line is truncated to the terminal width (fit by construction).

Reflowing content: instead of fixed lines, give the pager a
``reflow(width) -> lines`` callback and it re-renders whenever the
terminal width changes, keeping your place in the document. The
:meth:`PagerBuilder.markdown` shortcut does this for markdown, so text
re-wraps at word boundaries on every resize::

    PagerBuilder("README").markdown(Path("README.md").read_text()).run()

Custom ``on_key`` handlers let callers extend the pager: a handler may
end the run with ``PagerResult(key=...)`` so the caller knows *why* the
view closed and can act on it (e.g. jump to a related document).
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from termflow.ansi.codes import BOLD_ON, DIM_ON, RESET
from termflow.ansi.color import fg_color
from termflow.render.document import render_markdown_lines
from termflow.render.style import RenderFeatures, RenderStyle
from termflow.syntax import Highlighter
from termflow.tui.keys import Key, read_key
from termflow.tui.menu import RESIZE_POLL_S, _truncate
from termflow.tui.terminal import (
    CLEAR_TO_EOL,
    CURSOR_HOME,
    alt_screen,
    raw_mode,
    terminal_size,
)

_CHROME_LINES = 4  # title + blank above, blank + footer below


@dataclass
class PagerResult:
    """Outcome of a :meth:`Pager.run`.

    ``key`` is the key that closed the view (or the key a custom
    ``on_key`` handler ended the run with). ``cancelled`` is True for
    Esc / Ctrl+C, False for the ordinary ``q`` / Enter close.
    """

    key: str | None = None
    cancelled: bool = False


class Pager:
    """Interactive scrollable viewer. Prefer building via :class:`PagerBuilder`."""

    def __init__(
        self,
        title: str,
        *,
        lines: list[str] | None = None,
        text: str | None = None,
        style: RenderStyle | None = None,
        footer_hint: str | None = None,
        key_handlers: dict[str, Callable[[Pager], PagerResult | None]] | None = None,
        output: IO[str] | None = None,
        key_source: Callable[[], str] | None = None,
        size: Callable[[], tuple[int, int]] | None = None,
        use_alt_screen: bool = True,
        reflow: Callable[[int], list[str]] | None = None,
    ) -> None:
        if lines is None:
            lines = text.split("\n") if text is not None else []
        self._title = title
        self._lines = list(lines)
        self._reflow = reflow
        self._lines_width: int | None = None  # width `_lines` was rendered at
        self._style = style or RenderStyle.default()
        self._footer_hint = footer_hint
        self._key_handlers = dict(key_handlers or {})
        self._output = output if output is not None else sys.stdout
        self._read_key = key_source or (lambda: read_key(timeout=RESIZE_POLL_S))
        self._size = size or terminal_size
        self._use_alt_screen = use_alt_screen
        self._top = 0

    # -- state ---------------------------------------------------------------

    @property
    def top(self) -> int:
        """Index of the first visible content line."""
        return self._top

    @property
    def line_count(self) -> int:
        """Total number of content lines (at the current width)."""
        return len(self._content())

    def _content_width(self) -> int:
        # One column of right padding: writing the last column arms
        # deferred wrap and CLEAR_TO_EOL would erase the final char
        # (see Menu._frame).
        width, _ = self._size()
        return max(10, width - 1)

    def _content(self) -> list[str]:
        """Content lines, re-rendered first if reflowing and the width changed."""
        width = self._content_width()
        if self._reflow is not None and width != self._lines_width:
            self._replace_lines(self._reflow(width), width)
        return self._lines

    def _replace_lines(self, lines: list[str], width: int) -> None:
        old_count = len(self._lines)
        self._lines = lines
        self._lines_width = width
        # Keep the reader's place: same relative position in the document.
        if old_count:
            self._top = self._top * len(self._lines) // old_count
        self._top = max(0, min(self._top, self._max_top()))

    def _viewport(self) -> int:
        _, height = self._size()
        return max(1, height - _CHROME_LINES)

    def _max_top(self) -> int:
        return max(0, len(self._content()) - self._viewport())

    def scroll(self, delta: int) -> None:
        """Scroll by ``delta`` lines, clamped to the content bounds."""
        self._top = max(0, min(self._max_top(), self._top + delta))

    def _position_label(self) -> str:
        if self._max_top() == 0:
            return "All"
        if self._top == 0:
            return "Top"
        if self._top >= self._max_top():
            return "Bot"
        return f"{100 * self._top // self._max_top()}%"

    # -- painting ------------------------------------------------------------

    def _frame(self) -> list[str]:
        _, height = self._size()
        width = self._content_width()
        s = self._style
        viewport = self._viewport()
        body = self._content()[self._top : self._top + viewport]
        body += [""] * (viewport - len(body))
        hint = self._footer_hint if self._footer_hint is not None else "j/k scroll - q close"
        footer = f"{fg_color(s.grey)}{DIM_ON}{hint} - {self._position_label()}{RESET}"
        lines = [
            f"{fg_color(s.bright)}{BOLD_ON}{self._title}{RESET}",
            "",
            *body,
            "",
            footer,
        ]
        # Fit by construction: no frame line may exceed the terminal
        # width, ever -- wrapping would corrupt the repaint layout.
        return [_truncate(line, width) for line in lines[: max(1, height)]]

    def _paint(self) -> None:
        frame = self._frame()
        payload = CURSOR_HOME + "".join(f"{line}{CLEAR_TO_EOL}\r\n" for line in frame) + "\x1b[J"
        with contextlib.suppress(Exception):
            self._output.write(payload)
            self._output.flush()

    # -- event loop ----------------------------------------------------------

    def run(self) -> PagerResult:
        """Run the viewer until the user closes it."""
        if self._use_alt_screen:
            with raw_mode(), alt_screen(self._output):
                return self._loop()
        return self._loop()

    def _loop(self) -> PagerResult:
        while True:
            self._paint()
            result = self._handle_key(self._wait_key())
            if result is not None:
                return result

    def _wait_key(self) -> str:
        """Block for a key, repainting whenever the terminal resizes."""
        last_size = self._size()
        while True:
            key = self._read_key()
            if key:
                return key
            size = self._size()
            if size != last_size:
                last_size = size
                self._paint()

    def _handle_key(self, key: str) -> PagerResult | None:
        handler = self._key_handlers.get(key)
        if handler is not None:
            with contextlib.suppress(Exception):
                return handler(self)
            return None
        if key in ("q", Key.ENTER):
            return PagerResult(key=key)
        if key in (Key.ESCAPE, "ctrl-c"):
            return PagerResult(key=key, cancelled=True)
        if key in (Key.UP, "k"):
            self.scroll(-1)
        elif key in (Key.DOWN, "j"):
            self.scroll(1)
        elif key in (Key.PAGE_DOWN, " ", "f"):
            self.scroll(self._viewport())
        elif key in (Key.PAGE_UP, "b"):
            self.scroll(-self._viewport())
        elif key == "d":
            self.scroll(self._viewport() // 2)
        elif key == "u":
            self.scroll(-(self._viewport() // 2))
        elif key in (Key.HOME, "g"):
            self._top = 0
        elif key in (Key.END, "G"):
            self._top = self._max_top()
        return None


class PagerBuilder:
    """Fluent builder for :class:`Pager`."""

    def __init__(self, title: str) -> None:
        self._title = title
        self._kwargs: dict = {}
        self._markdown: dict | None = None

    def lines(self, lines: list[str]) -> PagerBuilder:
        self._kwargs["lines"] = lines
        return self

    def text(self, text: str) -> PagerBuilder:
        self._kwargs["text"] = text
        return self

    def reflow(self, reflow: Callable[[int], list[str]]) -> PagerBuilder:
        """Render content per width: ``reflow(width)`` is re-run on resize."""
        self._kwargs["reflow"] = reflow
        return self

    def markdown(
        self,
        markdown: str,
        *,
        features: RenderFeatures | None = None,
        highlighter: Highlighter | None = None,
        max_width: int | None = None,
    ) -> PagerBuilder:
        """Show rendered ``markdown``, re-wrapped to fit on every resize.

        Rendering uses the builder's :meth:`style`; ``max_width`` caps
        the wrap width on very wide terminals.
        """
        self._markdown = {
            "markdown": markdown,
            "features": features,
            "highlighter": highlighter,
            "max_width": max_width,
        }
        return self

    def style(self, style: RenderStyle) -> PagerBuilder:
        self._kwargs["style"] = style
        return self

    def footer_hint(self, hint: str) -> PagerBuilder:
        self._kwargs["footer_hint"] = hint
        return self

    def on_key(self, key: str, handler: Callable[[Pager], PagerResult | None]) -> PagerBuilder:
        self._kwargs.setdefault("key_handlers", {})[key] = handler
        return self

    def output(self, stream: IO[str]) -> PagerBuilder:
        self._kwargs["output"] = stream
        return self

    def key_source(self, source: Callable[[], str]) -> PagerBuilder:
        self._kwargs["key_source"] = source
        return self

    def size(self, size: Callable[[], tuple[int, int]]) -> PagerBuilder:
        self._kwargs["size"] = size
        return self

    def alt_screen(self, enabled: bool) -> PagerBuilder:
        self._kwargs["use_alt_screen"] = enabled
        return self

    def build(self) -> Pager:
        kwargs = dict(self._kwargs)
        if self._markdown is not None:
            kwargs["reflow"] = self._markdown_reflow(**self._markdown)
        return Pager(self._title, **kwargs)

    def _markdown_reflow(
        self,
        markdown: str,
        features: RenderFeatures | None,
        highlighter: Highlighter | None,
        max_width: int | None,
    ) -> Callable[[int], list[str]]:
        style = self._kwargs.get("style")
        highlighter = highlighter or Highlighter()  # shared across reflows

        def reflow(width: int) -> list[str]:
            width = min(width, max_width) if max_width else width
            return render_markdown_lines(markdown, width, style, features, highlighter)

        return reflow

    def run(self) -> PagerResult:
        return self.build().run()
