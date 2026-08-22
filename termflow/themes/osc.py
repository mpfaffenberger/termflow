"""Terminal-level palette swap via OSC escape sequences.

This module recolors the whole terminal window (bg, fg, ANSI palette
slots) using widely-supported xterm OSC sequences:

    OSC 10 ; spec BEL     -> set default foreground
    OSC 11 ; spec BEL     -> set default background
    OSC  4 ; N ; spec BEL -> set ANSI palette slot N (0..15)
    OSC 110 BEL           -> reset foreground
    OSC 111 BEL           -> reset background
    OSC 104 BEL           -> reset whole palette

Supported by iTerm2, Terminal.app, Alacritty, kitty, VS Code, GNOME
Terminal, Windows Terminal. Unsupported terminals silently ignore them.

An atexit handler is registered on first apply so the terminal is
restored when the process exits, even if the caller forgets to reset.
Persistence (replaying a palette next session) is deliberately the
caller's concern -- this module only talks to the live terminal.
"""

from __future__ import annotations

import atexit
import contextlib
import sys
from typing import IO

from termflow.themes.palette import TerminalPalette

BEL = "\007"
ESC = "\033"

_atexit_registered = False


# -- Low-level emit -----------------------------------------------------------
def _emit(seq: str, output: IO[str] | None = None) -> None:
    """Write an escape sequence, ignoring failures (closed tty etc.)."""
    stream = output if output is not None else sys.stdout
    try:
        stream.write(seq)
        stream.flush()
    except Exception:
        pass


def _osc(code: str, *args: str) -> str:
    """Build an OSC escape: ESC ] code ; args... BEL"""
    payload = ";".join((code, *args))
    return f"{ESC}]{payload}{BEL}"


# -- Public escape builders ---------------------------------------------------
def set_bg(color: str, output: IO[str] | None = None) -> None:
    """Set the terminal's default background color."""
    _emit(_osc("11", color), output)


def set_fg(color: str, output: IO[str] | None = None) -> None:
    """Set the terminal's default foreground color."""
    _emit(_osc("10", color), output)


def set_ansi_slot(slot: int, color: str, output: IO[str] | None = None) -> None:
    """Set one ANSI palette slot (0-15); out-of-range slots are ignored."""
    if not 0 <= slot <= 15:
        return
    _emit(_osc("4", str(slot), color), output)


def reset_bg(output: IO[str] | None = None) -> None:
    """Restore the terminal's original background color."""
    _emit(_osc("111"), output)


def reset_fg(output: IO[str] | None = None) -> None:
    """Restore the terminal's original foreground color."""
    _emit(_osc("110"), output)


def reset_ansi(output: IO[str] | None = None) -> None:
    """Restore the terminal's original 16-color ANSI palette."""
    _emit(_osc("104"), output)


# -- High-level API -----------------------------------------------------------
def apply_palette(
    palette: TerminalPalette | dict,
    output: IO[str] | None = None,
    register_reset: bool = True,
) -> None:
    """Apply a palette to the live terminal.

    Accepts either a :class:`TerminalPalette` or the raw dict shape
    (``{"bg": ..., "fg": ..., "ansi": [...]}``, all keys optional).

    ``register_reset=True`` ensures the terminal is restored at process
    exit. Pass ``output`` to target a stream other than ``sys.stdout``
    (mostly useful for tests).
    """
    if isinstance(palette, TerminalPalette):
        data = palette.to_dict()
    elif isinstance(palette, dict):
        data = palette
    else:
        return

    bg = data.get("bg")
    fg = data.get("fg")
    ansi = data.get("ansi") or []

    if bg:
        set_bg(bg, output)
    if fg:
        set_fg(fg, output)
    for i, color in enumerate(ansi[:16]):
        if color:
            set_ansi_slot(i, color, output)

    if register_reset:
        _ensure_atexit_registered()


def reset_palette(output: IO[str] | None = None) -> None:
    """Restore the terminal's original bg/fg/ANSI palette."""
    reset_ansi(output)
    reset_bg(output)
    reset_fg(output)


# -- Cleanup ------------------------------------------------------------------
def _at_exit_reset() -> None:
    """Best-effort terminal restore on Python exit."""
    with contextlib.suppress(Exception):
        reset_palette()


def _ensure_atexit_registered() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    try:
        atexit.register(_at_exit_reset)
        _atexit_registered = True
    except Exception:
        pass
