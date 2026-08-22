"""Keyboard input for TUI components.

Keys are plain strings: printable characters come through as themselves
(``"a"``, ``"?"``, ``" "``), everything else as a named constant from
:class:`Key` (``Key.UP``, ``Key.ENTER``, ``"ctrl-c"``...).

The escape-sequence parsing (:func:`parse_key`) is pure and testable;
:func:`read_key` layers blocking I/O on top (POSIX ``select``-based
escape disambiguation, ``msvcrt`` on Windows).
"""

from __future__ import annotations

import sys
from typing import IO

try:  # POSIX
    import select

    _HAS_SELECT = True
except ImportError:  # pragma: no cover - unlikely
    _HAS_SELECT = False

try:  # Windows
    import msvcrt  # type: ignore[import-not-found]

    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False


class Key:
    """Named keys returned by :func:`read_key` / :func:`parse_key`."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    ENTER = "enter"
    ESCAPE = "escape"
    BACKSPACE = "backspace"
    DELETE = "delete"
    TAB = "tab"
    HOME = "home"
    END = "end"
    PAGE_UP = "page-up"
    PAGE_DOWN = "page-down"


#: CSI / SS3 escape-sequence tails -> key names.
_ESCAPE_SEQUENCES: dict[str, str] = {
    "[A": Key.UP,
    "[B": Key.DOWN,
    "[C": Key.RIGHT,
    "[D": Key.LEFT,
    "[H": Key.HOME,
    "[F": Key.END,
    "[1~": Key.HOME,
    "[3~": Key.DELETE,
    "[4~": Key.END,
    "[5~": Key.PAGE_UP,
    "[6~": Key.PAGE_DOWN,
    "OA": Key.UP,
    "OB": Key.DOWN,
    "OC": Key.RIGHT,
    "OD": Key.LEFT,
    "OH": Key.HOME,
    "OF": Key.END,
}

#: Windows ``msvcrt`` two-byte scan codes -> key names.
_WINDOWS_SCAN_CODES: dict[str, str] = {
    "H": Key.UP,
    "P": Key.DOWN,
    "K": Key.LEFT,
    "M": Key.RIGHT,
    "G": Key.HOME,
    "O": Key.END,
    "I": Key.PAGE_UP,
    "Q": Key.PAGE_DOWN,
    "S": Key.DELETE,
}


def parse_key(char: str, escape_tail: str = "") -> str | None:
    """Translate a raw character (plus any escape tail) into a key name.

    Args:
        char: The first character read.
        escape_tail: Characters read after a leading ESC.

    Returns:
        A key name, a printable character, or None for unrecognized
        escape sequences (which callers should swallow).
    """
    if char == "\x1b":
        if not escape_tail:
            return Key.ESCAPE
        return _ESCAPE_SEQUENCES.get(escape_tail)
    if char in ("\r", "\n"):
        return Key.ENTER
    if char in ("\x7f", "\x08"):
        return Key.BACKSPACE
    if char == "\t":
        return Key.TAB
    if char < " ":
        # Remaining C0 control chars: ctrl-a .. ctrl-z and friends.
        code = ord(char)
        if 1 <= code <= 26:
            return f"ctrl-{chr(code + 96)}"
        return None
    return char


def _read_escape_tail(stream: IO[str], timeout: float) -> str:
    """Read the remainder of an escape sequence, if any bytes are pending."""
    tail = ""
    if not _HAS_SELECT:
        return tail
    try:
        fd = stream.fileno()
    except Exception:
        return tail
    # Escape sequences arrive as a burst; a lone ESC keypress does not.
    while select.select([fd], [], [], timeout)[0]:
        ch = stream.read(1)
        if not ch:
            break
        tail += ch
        # CSI sequences end with an alphabetic char or '~'.
        if tail[-1].isalpha() or tail[-1] == "~":
            break
        if len(tail) > 8:  # Defensive: never loop on garbage.
            break
    return tail


def _read_key_windows() -> str | None:  # pragma: no cover - Windows only
    ch = msvcrt.getwch()  # type: ignore[attr-defined]
    if ch in ("\x00", "\xe0"):
        return _WINDOWS_SCAN_CODES.get(msvcrt.getwch())  # type: ignore[attr-defined]
    return parse_key(ch)


def read_key(stream: IO[str] | None = None, escape_timeout: float = 0.02) -> str:
    """Block until a key is pressed and return its name.

    Unrecognized escape sequences are swallowed and the read retries,
    so callers always get something meaningful back.
    """
    stream = stream if stream is not None else sys.stdin
    while True:
        if _HAS_MSVCRT and stream is sys.stdin:  # pragma: no cover - Windows
            key = _read_key_windows()
        else:
            char = stream.read(1)
            if not char:
                return Key.ESCAPE  # EOF: treat as cancel.
            tail = _read_escape_tail(stream, escape_timeout) if char == "\x1b" else ""
            key = parse_key(char, tail)
        if key is not None:
            return key
