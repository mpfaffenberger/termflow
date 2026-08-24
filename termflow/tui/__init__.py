"""Dependency-free terminal UI toolkit.

Raw-ANSI building blocks for interactive menus: no curses, no
prompt_toolkit -- just escape sequences and small, injectable pieces.

* :mod:`termflow.tui.terminal` -- raw mode, alt screen, cursor control.
* :mod:`termflow.tui.keys` -- blocking key reads + escape parsing.
* :mod:`termflow.tui.menu` -- :class:`MenuBuilder` and friends.
* :mod:`termflow.tui.textinput` -- :class:`TextInputBuilder` line editing.
* :mod:`termflow.tui.pager` -- :class:`PagerBuilder` scrollable viewing.
* :mod:`termflow.tui.completion` -- a minimal completer protocol.
* :mod:`termflow.tui.layout` -- responsive split-pane composition.
"""

from termflow.tui.completion import (
    CompleteEvent,
    Completer,
    Completion,
    Document,
    merge_completers,
)
from termflow.tui.keys import Key, parse_key, read_key
from termflow.tui.layout import (
    COLLAPSE_BELOW,
    collapsed,
    split_frame,
    truncate,
    two_columns,
)
from termflow.tui.menu import Menu, MenuBuilder, MenuItem, MenuResult
from termflow.tui.pager import Pager, PagerBuilder, PagerResult
from termflow.tui.terminal import (
    alt_screen,
    raw_mode,
    terminal_session,
    terminal_size,
)
from termflow.tui.textinput import TextInput, TextInputBuilder, TextInputResult

__all__ = [
    "COLLAPSE_BELOW",
    "CompleteEvent",
    "Completer",
    "Completion",
    "Document",
    "Key",
    "Menu",
    "MenuBuilder",
    "MenuItem",
    "MenuResult",
    "Pager",
    "PagerBuilder",
    "PagerResult",
    "TextInput",
    "TextInputBuilder",
    "TextInputResult",
    "alt_screen",
    "collapsed",
    "merge_completers",
    "parse_key",
    "raw_mode",
    "read_key",
    "split_frame",
    "terminal_session",
    "terminal_size",
    "truncate",
    "two_columns",
]
