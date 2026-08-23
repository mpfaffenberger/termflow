"""Dependency-free terminal UI toolkit.

Raw-ANSI building blocks for interactive menus: no curses, no
prompt_toolkit -- just escape sequences and small, injectable pieces.

* :mod:`termflow.tui.terminal` -- raw mode, alt screen, cursor control.
* :mod:`termflow.tui.keys` -- blocking key reads + escape parsing.
* :mod:`termflow.tui.menu` -- :class:`MenuBuilder` and friends.
* :mod:`termflow.tui.textinput` -- :class:`TextInputBuilder` line editing.
"""

from termflow.tui.keys import Key, parse_key, read_key
from termflow.tui.menu import Menu, MenuBuilder, MenuItem, MenuResult
from termflow.tui.terminal import (
    alt_screen,
    raw_mode,
    terminal_session,
    terminal_size,
)
from termflow.tui.textinput import TextInput, TextInputBuilder, TextInputResult

__all__ = [
    "Key",
    "Menu",
    "MenuBuilder",
    "MenuItem",
    "MenuResult",
    "TextInput",
    "TextInputBuilder",
    "TextInputResult",
    "alt_screen",
    "parse_key",
    "raw_mode",
    "read_key",
    "terminal_session",
    "terminal_size",
]
