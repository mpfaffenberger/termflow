"""A declarative, dependency-free menu toolkit for terminal UIs.

Build interactive pickers with :class:`MenuBuilder`::

    from termflow.tui import MenuBuilder, MenuItem

    result = (
        MenuBuilder("Pick a model")
        .items([MenuItem("gpt-5", value="gpt-5", description="fast & smart")])
        .searchable()
        .page_size(10)
        .footer_hint("custom help text")  # optional
        .run()
    )
    if not result.cancelled:
        print(result.item.value)

Everything is injectable (key source, output stream, terminal size) so
menus are fully testable without a tty. Rendering is plain ANSI on the
alternate screen buffer -- no curses, no prompt_toolkit.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

from termflow.ansi.codes import BOLD_ON, DIM_ON, RESET
from termflow.ansi.color import fg_color
from termflow.render.style import RenderStyle
from termflow.tui.keys import Key, read_key
from termflow.tui.layout import collapsed
from termflow.tui.layout import truncate as _truncate
from termflow.tui.layout import two_columns as _two_columns
from termflow.tui.terminal import (
    CLEAR_TO_EOL,
    CURSOR_HOME,
    alt_screen,
    raw_mode,
    terminal_size,
)

#: How long one key wait may block before the loop rechecks the
#: terminal size. Resizes repaint within this interval at the latest.
RESIZE_POLL_S = 0.25

_POINTER = "> "
_NO_POINTER = "  "
_CHECKED = "◉ "
_UNCHECKED = "○ "


@dataclass
class MenuItem:
    """One selectable row.

    Attributes:
        label: Text shown in the list.
        value: Arbitrary payload returned on selection (defaults to label).
        description: Dim text shown after the label.
        disabled: Unselectable rows (section headers, separators).
    """

    label: str
    value: Any = None
    description: str = ""
    disabled: bool = False

    def __post_init__(self) -> None:
        if self.value is None:
            self.value = self.label


@dataclass
class MenuResult:
    """Outcome of a menu session.

    Attributes:
        cancelled: True when the user pressed Escape / ctrl-c.
        item: The highlighted item (single-select), or None.
        items: All toggled items (multi-select), else empty.
    """

    cancelled: bool = False
    item: MenuItem | None = None
    items: list[MenuItem] = field(default_factory=list)


class Menu:
    """Interactive list picker. Prefer building via :class:`MenuBuilder`."""

    def __init__(
        self,
        title: str,
        items: Sequence[MenuItem],
        *,
        style: RenderStyle | None = None,
        multi_select: bool = False,
        searchable: bool = False,
        page_size: int | None = None,
        preview: Callable[[MenuItem], str] | None = None,
        on_highlight: Callable[[MenuItem], None] | None = None,
        footer_hint: str | None = None,
        key_handlers: dict[str, Callable[[Menu, MenuItem], MenuResult | None]] | None = None,
        initial_index: int = 0,
        list_width: int | None = None,
        filter_fn: Callable[[str, MenuItem], bool] | None = None,
        output: IO[str] | None = None,
        key_source: Callable[[], str] | None = None,
        size: Callable[[], tuple[int, int]] | None = None,
        use_alt_screen: bool = True,
        inline: bool = False,
    ) -> None:
        self._title = title
        self._items = list(items)
        self._style = style or RenderStyle.default()
        self._multi = multi_select
        self._searchable = searchable
        self._page_size = max(1, page_size) if page_size is not None else None
        self._preview = preview
        self._on_highlight = on_highlight
        self._footer_hint = footer_hint
        self._key_handlers = dict(key_handlers or {})
        self._list_width = list_width
        self._filter_fn = filter_fn
        self._output = output if output is not None else sys.stdout
        self._read_key = key_source or (lambda: read_key(timeout=RESIZE_POLL_S))
        self._size = size or terminal_size
        self._use_alt_screen = use_alt_screen and not inline
        self._inline = inline
        self._painted_lines = 0

        self._cursor = max(0, min(initial_index, max(len(self._items) - 1, 0)))
        self._search = ""
        self._checked: set[int] = set()  # indexes into self._items

    @property
    def highlighted(self) -> MenuItem | None:
        """The currently highlighted item (None when filtered empty)."""
        rows = self._filtered()
        self._clamp_cursor(rows)
        return rows[self._cursor][1] if rows else None

    def replace_items(self, items: Sequence[MenuItem]) -> None:
        """Swap the menu rows in place (for key handlers that mutate state)."""
        self._items = list(items)
        self._checked.clear()
        self._clamp_cursor(self._filtered())

    def clear_search(self) -> None:
        """Reset the search filter (for a bound clear-filter key)."""
        self._search = ""

    def _effective_page_size(self) -> int:
        """Fixed page size if configured, else fit the terminal height."""
        if self._page_size is not None:
            return self._page_size
        _cols, rows_avail = self._size()
        # Overhead: title + optional search row + blank + blank + footer.
        overhead = 4 + (1 if self._searchable else 0)
        return max(1, rows_avail - overhead)

    def page_up(self) -> None:
        """Move the cursor one page toward the top."""
        self._cursor = max(0, self._cursor - self._effective_page_size())

    def page_down(self) -> None:
        """Move the cursor one page toward the bottom."""
        rows = self._filtered()
        self._cursor = min(max(len(rows) - 1, 0), self._cursor + self._effective_page_size())

    # -- state helpers ------------------------------------------------------
    def _filtered(self) -> list[tuple[int, MenuItem]]:
        """(original_index, item) pairs matching the search filter."""
        if not self._search:
            return list(enumerate(self._items))
        if self._filter_fn is not None:
            matches = self._filter_fn
        else:

            def matches(query: str, item: MenuItem) -> bool:
                return query.lower() in item.label.lower()

        return [(i, it) for i, it in enumerate(self._items) if matches(self._search, it)]

    def _move_cursor(self, rows: list[tuple[int, MenuItem]], delta: int) -> None:
        """Move the cursor by delta, wrapping and skipping disabled rows."""
        if not rows:
            return
        n = len(rows)
        pos = self._cursor
        for _ in range(n):
            pos = (pos + delta) % n
            if not rows[pos][1].disabled:
                self._cursor = pos
                self._fire_highlight(rows)
                return

    def _fire_highlight(self, rows: list[tuple[int, MenuItem]]) -> None:
        if self._on_highlight and rows:
            with contextlib.suppress(Exception):
                self._on_highlight(rows[self._cursor][1])

    def _clamp_cursor(self, rows: list[tuple[int, MenuItem]]) -> None:
        self._cursor = max(0, min(self._cursor, len(rows) - 1)) if rows else 0

    # -- rendering ----------------------------------------------------------
    def _footer(self, rows: list[tuple[int, MenuItem]]) -> str:
        if self._footer_hint is not None:
            return self._footer_hint
        parts = ["↑/↓ move", "enter select"]
        if self._multi:
            parts.insert(1, "space toggle")
        if self._searchable:
            parts.append("type to filter")
        parts.append("esc cancel")
        page_size = self._effective_page_size()
        pages = (len(rows) + page_size - 1) // page_size
        if pages > 1:
            parts.append(f"page {self._cursor // page_size + 1}/{pages}")
        return " · ".join(parts)

    def _render_row(self, pos: int, index: int, item: MenuItem, width: int) -> str:
        s = self._style
        if item.disabled:
            return f"{DIM_ON}{_NO_POINTER}{item.label}{RESET}"
        pointer = _POINTER if pos == self._cursor else _NO_POINTER
        check = ""
        if self._multi:
            check = _CHECKED if index in self._checked else _UNCHECKED
        label = item.label
        desc = f"  {DIM_ON}{item.description}{RESET}" if item.description else ""
        if pos == self._cursor:
            line = f"{fg_color(s.bright)}{BOLD_ON}{pointer}{check}{label}{RESET}{desc}"
        else:
            line = f"{pointer}{check}{label}{desc}"
        return _truncate(line, width)

    def _frame(self) -> list[str]:
        cols, rows_avail = self._size()
        # Paint one column short of the terminal edge. Writing the last
        # column arms xterm's deferred-wrap state, where the trailing
        # clear-to-EOL erases the character just written -- full-width
        # lines would lose their final char on real terminals.
        cols = max(10, cols - 1)
        rows = self._filtered()
        self._clamp_cursor(rows)
        s = self._style

        lines = [f"{fg_color(s.bright)}{BOLD_ON}{self._title}{RESET}"]
        if self._searchable:
            hint = self._search or "(type to filter)"
            style_on = "" if self._search else DIM_ON
            lines.append(f"{fg_color(s.symbol)}search:{RESET} {style_on}{hint}{RESET}")
        lines.append("")

        page_size = self._effective_page_size()
        page_start = (self._cursor // page_size) * page_size
        page = rows[page_start : page_start + page_size]
        # Narrow terminals collapse the preview pane entirely: the list
        # takes the full width (see termflow.tui.layout).
        show_preview = self._preview is not None and not collapsed(cols)
        list_width = (self._list_width or max(20, cols // 2)) if show_preview else cols
        body = [
            self._render_row(page_start + offset, index, item, list_width - 1)
            for offset, (index, item) in enumerate(page)
        ]
        if not body:
            body = [f"{DIM_ON}(no matches){RESET}"]

        if show_preview and rows:
            preview_text = self._preview_text(rows[self._cursor][1])
            body = _two_columns(body, preview_text.splitlines(), list_width, cols)

        # Clamp to the terminal height (header + body + blank + footer):
        # a preview taller than the screen must not push the list and
        # title into scrollback -- clip the body, keep the footer.
        body_budget = max(1, rows_avail - len(lines) - 2)
        if len(body) > body_budget:
            body = body[:body_budget]

        lines.extend(body)
        lines.append("")
        lines.append(f"{DIM_ON}{self._footer(rows)}{RESET}")
        return lines

    def _preview_text(self, item: MenuItem) -> str:
        try:
            return self._preview(item) if self._preview else ""
        except Exception:
            return ""

    def _paint(self) -> None:
        # Full repaint: home the cursor (or, inline, climb back to the
        # first painted row), redraw every line with clear-to-eol, then
        # clear anything below the frame.
        frame = self._frame()
        if self._inline:
            home = f"\r\x1b[{self._painted_lines}A" if self._painted_lines else "\r"
            self._painted_lines = len(frame)
        else:
            home = CURSOR_HOME
        payload = home + "".join(f"{line}{CLEAR_TO_EOL}\r\n" for line in frame) + "\x1b[J"
        try:
            self._output.write(payload)
            self._output.flush()
        except Exception:
            pass

    # -- event loop ---------------------------------------------------------
    def run(self) -> MenuResult:
        """Run the menu until the user selects or cancels.

        Inline menus paint below the current cursor position (inquirer
        style) and repaint in place, leaving the transcript above them
        untouched; the final frame scrolls into history on exit.
        """
        if self._use_alt_screen:
            with raw_mode(), alt_screen(self._output):
                return self._loop()
        if self._inline:
            with raw_mode():
                return self._loop()
        return self._loop()

    def _loop(self) -> MenuResult:
        rows = self._filtered()
        if rows and rows[self._cursor][1].disabled:
            self._move_cursor(rows, 1)
        self._fire_highlight(rows)
        while True:
            self._paint()
            key = self._wait_key()
            rows = self._filtered()
            self._clamp_cursor(rows)
            result = self._handle_key(key, rows)
            if result is not None:
                return result

    def _wait_key(self) -> str:
        """Block for a key, repainting whenever the terminal resizes.

        The default key source times out every :data:`RESIZE_POLL_S`
        seconds (returning ``""``), giving this loop a chance to notice
        a size change and repaint without any keypress. Injected test
        sources that never return ``""`` behave exactly as before.
        """
        last_size = self._size()
        while True:
            key = self._read_key()
            if key:
                return key
            size = self._size()
            if size != last_size:
                last_size = size
                self._paint()

    def _handle_key(self, key: str, rows: list[tuple[int, MenuItem]]) -> MenuResult | None:
        handler = self._key_handlers.get(key)
        if handler is not None and rows:
            with contextlib.suppress(Exception):
                return handler(self, rows[self._cursor][1])
            return None
        if key in (Key.ESCAPE, "ctrl-c"):
            return MenuResult(cancelled=True)
        if key == Key.ENTER:
            return self._select(rows)
        if key == Key.UP:
            self._move_cursor(rows, -1)
        elif key == Key.DOWN:
            self._move_cursor(rows, 1)
        elif key == Key.PAGE_UP:
            self.page_up()
            self._fire_highlight(rows)
        elif key == Key.PAGE_DOWN:
            self.page_down()
            self._fire_highlight(rows)
        elif key == Key.HOME:
            self._cursor = 0
            self._fire_highlight(rows)
        elif key == Key.END:
            self._cursor = max(len(rows) - 1, 0)
            self._fire_highlight(rows)
        elif key == " " and self._multi and rows:
            index = rows[self._cursor][0]
            self._checked.symmetric_difference_update({index})
        elif key == Key.BACKSPACE and self._searchable:
            self._search = self._search[:-1]
        elif self._searchable and len(key) == 1 and key.isprintable():
            self._search += key
            self._cursor = 0
        return None

    def _select(self, rows: list[tuple[int, MenuItem]]) -> MenuResult | None:
        if not rows:
            return None
        highlighted = rows[self._cursor][1]
        if highlighted.disabled:
            return None
        if self._multi:
            # Enter with nothing toggled selects the highlighted row.
            chosen = [self._items[i] for i in sorted(self._checked)] or [highlighted]
            return MenuResult(item=highlighted, items=chosen)
        return MenuResult(item=highlighted, items=[highlighted])


class MenuBuilder:
    """Fluent builder for :class:`Menu`. Each setter returns ``self``."""

    def __init__(self, title: str) -> None:
        self._title = title
        self._kwargs: dict[str, Any] = {}
        self._items: list[MenuItem] = []

    def items(self, items: Iterable[MenuItem | str]) -> MenuBuilder:
        """Set the menu rows; bare strings become simple MenuItems."""
        self._items = [it if isinstance(it, MenuItem) else MenuItem(it) for it in items]
        return self

    def style(self, style: RenderStyle) -> MenuBuilder:
        self._kwargs["style"] = style
        return self

    def multi_select(self, enabled: bool = True) -> MenuBuilder:
        self._kwargs["multi_select"] = enabled
        return self

    def searchable(self, enabled: bool = True) -> MenuBuilder:
        self._kwargs["searchable"] = enabled
        return self

    def page_size(self, size: int | None) -> MenuBuilder:
        """Fix rows per page; default (None) auto-fits the terminal height."""
        self._kwargs["page_size"] = size
        return self

    def preview(self, callback: Callable[[MenuItem], str]) -> MenuBuilder:
        """Right-hand preview pane fed by the highlighted item."""
        self._kwargs["preview"] = callback
        return self

    def on_highlight(self, callback: Callable[[MenuItem], None]) -> MenuBuilder:
        """Fire on every cursor move (live theme previews, etc.)."""
        self._kwargs["on_highlight"] = callback
        return self

    def footer_hint(self, text: str) -> MenuBuilder:
        self._kwargs["footer_hint"] = text
        return self

    def on_key(
        self, key: str, handler: Callable[[Menu, MenuItem], MenuResult | None]
    ) -> MenuBuilder:
        """Bind a custom action key (e.g. ``"p"`` to pin, ``"d"`` to delete).

        The handler receives the running :class:`Menu` and the highlighted
        item. Return a :class:`MenuResult` to exit the menu with it, or
        ``None`` to repaint and keep going. Custom keys take precedence
        over built-in handling (including search typing).
        """
        self._kwargs.setdefault("key_handlers", {})[key] = handler
        return self

    def initial_index(self, index: int) -> MenuBuilder:
        """Open with the cursor on this item index (e.g. current selection)."""
        self._kwargs["initial_index"] = index
        return self

    def list_width(self, width: int) -> MenuBuilder:
        """Fix the left column width when a preview pane is present."""
        self._kwargs["list_width"] = width
        return self

    def filter_fn(self, matches: Callable[[str, MenuItem], bool]) -> MenuBuilder:
        """Replace the default substring search with a custom matcher."""
        self._kwargs["filter_fn"] = matches
        return self

    def output(self, stream: IO[str]) -> MenuBuilder:
        self._kwargs["output"] = stream
        return self

    def key_source(self, source: Callable[[], str]) -> MenuBuilder:
        self._kwargs["key_source"] = source
        return self

    def size(self, size: Callable[[], tuple[int, int]]) -> MenuBuilder:
        self._kwargs["size"] = size
        return self

    def alt_screen(self, enabled: bool = True) -> MenuBuilder:
        self._kwargs["use_alt_screen"] = enabled
        return self

    def inline(self, enabled: bool = True) -> MenuBuilder:
        """Paint at the cursor (inquirer style) instead of owning the screen."""
        self._kwargs["inline"] = enabled
        return self

    def build(self) -> Menu:
        return Menu(self._title, self._items, **self._kwargs)

    def run(self) -> MenuResult:
        """Convenience: build and run in one call."""
        return self.build().run()
