"""Terminal session for live (real-time) mode.

:class:`LiveConsole` owns the terminal while an app runs: alternate
screen, hidden cursor, autowrap off (so painting the last column never
scrolls), synchronized-output frames, and a platform input backend:

* Windows -> :class:`termflow.live._win32.WindowsConsoleIO`
  (native press/release records; works in Windows Terminal and conhost)
* POSIX   -> :class:`PosixConsoleIO` (raw termios + kitty keyboard
  protocol when the terminal supports it)

Everything restores on exit, exceptions included.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import IO, Protocol

from termflow.live.input import KeyEvent, VTInputParser
from termflow.tui.terminal import (
    ALT_SCREEN_OFF,
    ALT_SCREEN_ON,
    CLEAR_SCREEN,
    CURSOR_HIDE,
    CURSOR_HOME,
    CURSOR_SHOW,
)

AUTOWRAP_OFF = "\x1b[?7l"
AUTOWRAP_ON = "\x1b[?7h"
SYNC_BEGIN = "\x1b[?2026h"  # DEC mode 2026: paint the frame atomically (no tearing)
SYNC_END = "\x1b[?2026l"
#: Kitty keyboard protocol: push flags 1|2|8 (disambiguate, report
#: event types, all keys as escape codes) and query support.
KITTY_PUSH = "\x1b[>11u\x1b[?u"
KITTY_POP = "\x1b[<u"


class ConsoleIO(Protocol):
    """A platform input backend."""

    @property
    def reports_release(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def poll(self) -> list[KeyEvent]: ...


def _write(stream: IO[str], text: str) -> None:
    with contextlib.suppress(Exception):
        stream.write(text)
        stream.flush()


class PosixConsoleIO:  # pragma: no cover - needs a real tty
    """Raw-mode stdin reader with kitty keyboard protocol negotiation."""

    def __init__(self, output: IO[str], escape_timeout: float = 0.02) -> None:
        self._output = output
        self._escape_timeout = escape_timeout
        self._parser = VTInputParser()
        self._fd = sys.stdin.fileno()
        self._saved: list | None = None

    @property
    def reports_release(self) -> bool:
        return self._parser.kitty

    def start(self) -> None:
        if sys.platform == "win32":
            raise RuntimeError("use termflow.live._win32.WindowsConsoleIO on Windows")
        import termios
        import tty

        self._saved = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        _write(self._output, KITTY_PUSH)

    def stop(self) -> None:
        if sys.platform == "win32":
            return
        import termios

        _write(self._output, KITTY_POP)
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def _read_available(self, timeout: float = 0.0) -> str:
        import select

        chunks = []
        while select.select([self._fd], [], [], timeout)[0]:
            data = os.read(self._fd, 4096)
            if not data:
                break
            chunks.append(data)
            timeout = 0.0
        return b"".join(chunks).decode("utf-8", errors="replace")

    def poll(self) -> list[KeyEvent]:
        events = self._parser.feed(self._read_available())
        if self._parser.pending:
            # Lone ESC vs. the start of a sequence: give the burst a moment.
            events += self._parser.feed(self._read_available(self._escape_timeout))
            if self._parser.pending:
                events += self._parser.flush()
        return events


def default_console_io(output: IO[str]) -> ConsoleIO:  # pragma: no cover - platform glue
    """The right input backend for this platform."""
    if sys.platform == "win32":
        from termflow.live._win32 import WindowsConsoleIO

        return WindowsConsoleIO()
    return PosixConsoleIO(output)


class LiveConsole:
    """Context manager that turns the terminal into a frame display.

    Args:
        output: Where frames go (stdout by default).
        io: Input backend (platform default when None). Inject a fake
            one in tests.
    """

    def __init__(self, output: IO[str] | None = None, io: ConsoleIO | None = None) -> None:
        self.output = output if output is not None else sys.stdout
        self._io = io
        self.bytes_written = 0

    @property
    def reports_release(self) -> bool:
        return self._io is not None and self._io.reports_release

    def __enter__(self) -> LiveConsole:
        if self._io is None:
            self._io = default_console_io(self.output)
        # Alt screen first: kitty keeps separate flag stacks per screen.
        _write(self.output, ALT_SCREEN_ON + CLEAR_SCREEN + CURSOR_HOME + CURSOR_HIDE + AUTOWRAP_OFF)
        try:
            self._io.start()
        except BaseException:
            _write(self.output, AUTOWRAP_ON + CURSOR_SHOW + ALT_SCREEN_OFF)
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            if self._io is not None:
                self._io.stop()
        finally:
            _write(self.output, "\x1b[0m" + AUTOWRAP_ON + CURSOR_SHOW + ALT_SCREEN_OFF)

    def poll(self) -> list[KeyEvent]:
        return self._io.poll() if self._io is not None else []

    def present(self, frame: str) -> None:
        """Write one frame atomically (no-op for empty diffs)."""
        if frame:
            data = SYNC_BEGIN + frame + SYNC_END
            self.bytes_written += len(data)
            _write(self.output, data)
