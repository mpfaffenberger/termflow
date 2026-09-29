"""Terminal-level theming for termflow.

Two layers:

* :class:`TerminalPalette` + :data:`PALETTES` -- bundled 16-color
  terminal palettes (bg, fg, ANSI slots 0-15) with a bridge to
  :class:`~termflow.render.style.RenderStyle`.
* :mod:`termflow.themes.osc` -- applies palettes to the live terminal
  via xterm OSC escape sequences (OSC 4/10/11), with best-effort
  restore at process exit.
"""

from termflow.themes.osc import (
    apply_palette,
    reset_palette,
    set_ansi_slot,
    set_bg,
    set_fg,
)
from termflow.themes.palette import (
    PALETTES,
    TerminalPalette,
    get_palette,
    palette_names,
)

__all__ = [
    "PALETTES",
    "TerminalPalette",
    "apply_palette",
    "get_palette",
    "palette_names",
    "reset_palette",
    "set_ansi_slot",
    "set_bg",
    "set_fg",
]
