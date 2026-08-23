"""A single-line text input widget for terminal UIs.

Build line editors with :class:`TextInputBuilder`::

    from termflow.tui import TextInputBuilder

    result = (
        TextInputBuilder("Add model")
        .prompt("API key: ")
        .placeholder("sk-...")
        .mask("*")
        .validator(lambda text: None if text else "a key is required")
        .run()
    )
    if not result.cancelled:
        print(result.value)

Everything is injectable (key source, output stream, terminal size) so
inputs are fully testable without a tty -- the same contract as
:class:`termflow.tui.menu.Menu`. Editing supports cursor movement
(arrows, Home/End, Ctrl+A/E), deletion (Backspace, Delete, Ctrl+U/K/W),
placeholder text, masked entry, validation on commit, and horizontal
scrolling for values wider than the terminal.

Custom ``on_key`` handlers let callers compose multi-field forms: a
handler may end the run with ``TextInputResult(key="down")`` so the
caller knows *why* editing stopped and can move focus accordingly.
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
from termflow.ansi.utils import visible_length
from termflow.render.style import RenderStyle
from termflow.tui.keys import Key, read_key
from termflow.tui.menu import RESIZE_POLL_S, _truncate
from termflow.tui.terminal import (
    CLEAR_TO_EOL,
    CURSOR_HOME,
    alt_screen,
    raw_mode,
    terminal_size,
)

_REVERSE_ON = "\x1b[7m"
_REVERSE_OFF = "\x1b[27m"
_WORD_SEPARATORS = " \t/-_.:,;"


@dataclass
class TextInputResult:
    """Outcome of a :meth:`TextInput.run`.

    ``value`` is the committed text (``None`` when cancelled). ``key``
    is set when a custom ``on_key`` handler ended the run, so form
    callers can tell an Enter-commit from a focus-move.
    """

    value: str | None = None
    cancelled: bool = False
    key: str | None = None


class TextInput:
    """Interactive line editor. Prefer building via :class:`TextInputBuilder`."""

    def __init__(
        self,
        title: str,
        *,
        prompt: str = "> ",
        initial: str = "",
        placeholder: str = "",
        mask: str | None = None,
        validator: Callable[[str], str | None] | None = None,
        style: RenderStyle | None = None,
        footer_hint: str | None = None,
        key_handlers: dict[str, Callable[[TextInput], TextInputResult | None]] | None = None,
        output: IO[str] | None = None,
        key_source: Callable[[], str] | None = None,
        size: Callable[[], tuple[int, int]] | None = None,
        use_alt_screen: bool = True,
    ) -> None:
        self._title = title
        self._prompt = prompt
        self._placeholder = placeholder
        self._mask = mask
        self._validator = validator
        self._style = style or RenderStyle.default()
        self._footer_hint = footer_hint
        self._key_handlers = dict(key_handlers or {})
        self._output = output if output is not None else sys.stdout
        self._read_key = key_source or (lambda: read_key(timeout=RESIZE_POLL_S))
        self._size = size or terminal_size
        self._use_alt_screen = use_alt_screen

        self._chars: list[str] = list(initial)
        self._cursor = len(self._chars)
        self._error: str | None = None
        self._scroll = 0

    # -- state ---------------------------------------------------------------

    @property
    def text(self) -> str:
        """The current (uncommitted) buffer contents."""
        return "".join(self._chars)

    @property
    def cursor(self) -> int:
        """Cursor position in codepoints (0 .. len(text))."""
        return self._cursor

    def set_text(self, text: str) -> None:
        """Replace the buffer and park the cursor at the end."""
        self._chars = list(text)
        self._cursor = len(self._chars)
        self._error = None

    # -- editing -------------------------------------------------------------

    def _insert(self, char: str) -> None:
        self._chars.insert(self._cursor, char)
        self._cursor += 1
        self._error = None

    def _backspace(self) -> None:
        if self._cursor > 0:
            del self._chars[self._cursor - 1]
            self._cursor -= 1
            self._error = None

    def _delete(self) -> None:
        if self._cursor < len(self._chars):
            del self._chars[self._cursor]
            self._error = None

    def _kill_to_start(self) -> None:
        del self._chars[: self._cursor]
        self._cursor = 0
        self._error = None

    def _kill_to_end(self) -> None:
        del self._chars[self._cursor :]
        self._error = None

    def _delete_word_back(self) -> None:
        end = self._cursor
        pos = end
        while pos > 0 and self._chars[pos - 1] in _WORD_SEPARATORS:
            pos -= 1
        while pos > 0 and self._chars[pos - 1] not in _WORD_SEPARATORS:
            pos -= 1
        if pos < end:
            del self._chars[pos:end]
            self._cursor = pos
            self._error = None

    # -- rendering -----------------------------------------------------------

    def _display_chars(self) -> list[str]:
        if self._mask:
            return [self._mask[0]] * len(self._chars)
        return self._chars

    def _keep_cursor_visible(self, available: int) -> None:
        """Adjust the scroll window so the cursor cell stays on screen."""
        chars = self._display_chars()
        if self._cursor < self._scroll:
            self._scroll = self._cursor
        while True:
            used = sum(visible_length(c) for c in chars[self._scroll : self._cursor])
            if used < available or self._scroll >= self._cursor:
                break
            self._scroll += 1

    def _input_line(self, width: int) -> str:
        s = self._style
        prompt = f"{fg_color(s.symbol)}{self._prompt}{RESET}"
        available = max(4, width - visible_length(self._prompt) - 1)

        if not self._chars and self._placeholder:
            ghost = self._placeholder[: available - 1]
            return f"{prompt}{_REVERSE_ON} {_REVERSE_OFF}{DIM_ON}{ghost}{RESET}"

        self._keep_cursor_visible(available)
        chars = self._display_chars()
        cells: list[str] = []
        used = 0
        cursor_drawn = False
        for index in range(self._scroll, len(chars) + 1):
            at_cursor = index == self._cursor
            char = chars[index] if index < len(chars) else " "
            cell_width = visible_length(char)
            if used + cell_width > available:
                break
            if at_cursor:
                cells.append(f"{_REVERSE_ON}{char}{_REVERSE_OFF}")
                cursor_drawn = True
            elif index < len(chars):
                cells.append(char)
            else:
                break
            used += cell_width
        body = "".join(cells)
        if not cursor_drawn and self._cursor < self._scroll:
            body = f"{_REVERSE_ON} {_REVERSE_OFF}{body}"
        return f"{prompt}{body}"

    def _frame(self) -> list[str]:
        width, height = self._size()
        s = self._style
        lines = [
            f"{fg_color(s.bright)}{BOLD_ON}{self._title}{RESET}",
            "",
            self._input_line(width),
        ]
        if self._error:
            lines.append(f"{fg_color(s.error)}{self._error}{RESET}")
        hint = self._footer_hint if self._footer_hint is not None else "Enter accept - Esc cancel"
        if hint:
            lines.extend(["", f"{fg_color(s.grey)}{DIM_ON}{hint}{RESET}"])
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

    def run(self) -> TextInputResult:
        """Run the editor until the user commits or cancels."""
        if self._use_alt_screen:
            with raw_mode(), alt_screen(self._output):
                return self._loop()
        return self._loop()

    def _loop(self) -> TextInputResult:
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

    def _commit(self) -> TextInputResult | None:
        if self._validator is not None:
            self._error = self._validator(self.text)
            if self._error is not None:
                return None
        return TextInputResult(value=self.text)

    def _handle_key(self, key: str) -> TextInputResult | None:
        handler = self._key_handlers.get(key)
        if handler is not None:
            with contextlib.suppress(Exception):
                return handler(self)
            return None
        if key == Key.ENTER:
            return self._commit()
        if key in (Key.ESCAPE, "ctrl-c"):
            return TextInputResult(cancelled=True)
        if key in (Key.LEFT, "ctrl-b"):
            self._cursor = max(0, self._cursor - 1)
        elif key in (Key.RIGHT, "ctrl-f"):
            self._cursor = min(len(self._chars), self._cursor + 1)
        elif key in (Key.HOME, "ctrl-a"):
            self._cursor = 0
        elif key in (Key.END, "ctrl-e"):
            self._cursor = len(self._chars)
        elif key == Key.BACKSPACE:
            self._backspace()
        elif key == Key.DELETE:
            self._delete()
        elif key == "ctrl-u":
            self._kill_to_start()
        elif key == "ctrl-k":
            self._kill_to_end()
        elif key == "ctrl-w":
            self._delete_word_back()
        elif len(key) == 1 and key.isprintable():
            self._insert(key)
        return None


class TextInputBuilder:
    """Fluent builder for :class:`TextInput`."""

    def __init__(self, title: str) -> None:
        self._title = title
        self._kwargs: dict = {}

    def prompt(self, text: str) -> TextInputBuilder:
        self._kwargs["prompt"] = text
        return self

    def initial(self, text: str) -> TextInputBuilder:
        self._kwargs["initial"] = text
        return self

    def placeholder(self, text: str) -> TextInputBuilder:
        self._kwargs["placeholder"] = text
        return self

    def mask(self, char: str = "*") -> TextInputBuilder:
        self._kwargs["mask"] = char
        return self

    def validator(self, check: Callable[[str], str | None]) -> TextInputBuilder:
        self._kwargs["validator"] = check
        return self

    def style(self, style: RenderStyle) -> TextInputBuilder:
        self._kwargs["style"] = style
        return self

    def footer_hint(self, text: str) -> TextInputBuilder:
        self._kwargs["footer_hint"] = text
        return self

    def on_key(
        self,
        key: str,
        handler: Callable[[TextInput], TextInputResult | None],
    ) -> TextInputBuilder:
        self._kwargs.setdefault("key_handlers", {})[key] = handler
        return self

    def output(self, stream: IO[str]) -> TextInputBuilder:
        self._kwargs["output"] = stream
        return self

    def key_source(self, source: Callable[[], str]) -> TextInputBuilder:
        self._kwargs["key_source"] = source
        return self

    def size(self, size: Callable[[], tuple[int, int]]) -> TextInputBuilder:
        self._kwargs["size"] = size
        return self

    def alt_screen(self, enabled: bool = True) -> TextInputBuilder:
        self._kwargs["use_alt_screen"] = enabled
        return self

    def build(self) -> TextInput:
        return TextInput(self._title, **self._kwargs)

    def run(self) -> TextInputResult:
        return self.build().run()
