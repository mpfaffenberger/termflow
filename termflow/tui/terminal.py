"""Low-level terminal control for interactive TUI components.

Raw mode, alternate screen, and cursor visibility — the three
ingredients every full-screen menu needs. POSIX gets real ``termios``
raw mode; on Windows raw mode is a no-op because ``msvcrt`` key reads
(see :mod:`termflow.tui.keys`) never echo in the first place. Modern
Windows terminals honor the VT alternate-screen/cursor sequences.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

try:  # POSIX
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover - Windows
    _HAS_TERMIOS = False

# VT control sequences
ALT_SCREEN_ON = "\x1b[?1049h"
ALT_SCREEN_OFF = "\x1b[?1049l"
CURSOR_HIDE = "\x1b[?25l"
CURSOR_SHOW = "\x1b[?25h"
CURSOR_HOME = "\x1b[H"
CLEAR_SCREEN = "\x1b[2J"
CLEAR_TO_EOL = "\x1b[K"


def terminal_size(fallback: tuple[int, int] = (80, 24)) -> tuple[int, int]:
    """Return (columns, rows) of the terminal, with a sane fallback."""
    size = shutil.get_terminal_size(fallback=fallback)
    return size.columns, size.lines


def _write(stream: IO[str], text: str) -> None:
    """Write + flush, swallowing failures on closed/odd streams."""
    try:
        stream.write(text)
        stream.flush()
    except Exception:
        pass


@contextlib.contextmanager
def raw_mode(stream: IO[str] | None = None) -> Iterator[None]:
    """Put ``stream`` (default stdin) into raw mode for the duration.

    No-ops gracefully when the stream is not a tty (pipes, tests) or on
    platforms without ``termios``.
    """
    stream = stream if stream is not None else sys.stdin
    if not _HAS_TERMIOS:
        yield
        return
    try:
        fd = stream.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        yield
        return
    try:
        tty.setraw(fd)
        yield
    finally:
        with contextlib.suppress(Exception):
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


@contextlib.contextmanager
def alt_screen(output: IO[str] | None = None, hide_cursor: bool = True) -> Iterator[None]:
    """Switch to the alternate screen buffer (like vim/less) for the duration.

    Restores the primary screen and cursor visibility on exit, even on
    exceptions, so a crashing menu never wrecks the user's scrollback.
    """
    stream = output if output is not None else sys.stdout
    _write(stream, ALT_SCREEN_ON + CLEAR_SCREEN + CURSOR_HOME)
    if hide_cursor:
        _write(stream, CURSOR_HIDE)
    try:
        yield
    finally:
        if hide_cursor:
            _write(stream, CURSOR_SHOW)
        _write(stream, ALT_SCREEN_OFF)
