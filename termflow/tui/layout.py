"""Responsive split-pane layout helpers.

The master-detail pattern for terminal UIs: two panes side by side on
wide terminals, one focused pane at a time on narrow ones (phones,
split tmux panes). Pure functions -- widgets decide *what* to draw,
:func:`split_frame` decides *how* it fits::

    from termflow.tui.layout import collapsed, split_frame

    frame = split_frame(
        left_pane_lines,
        right_pane_lines,
        width=cols,
        list_width=30,
        focus="right" if picking_detail else "left",
    )

When ``collapsed(width)`` is true, callers should also adapt their key
hints (e.g. advertise "back" instead of pane-switching) -- the layout
only handles geometry.

:func:`truncate` and :func:`two_columns` are the underlying primitives,
shared with :class:`termflow.tui.menu.Menu`.
"""

from __future__ import annotations

from typing import Literal

from termflow.ansi.codes import DIM_ON, RESET
from termflow.ansi.utils import visible_length

#: Below this many columns, split layouts collapse to a single pane.
#: Chosen so a canonical 80-column terminal (79 after the one-column
#: right padding) keeps its side-by-side layout, while phone-sized
#: panes collapse to the two-phase master/detail flow.
COLLAPSE_BELOW = 76


def collapsed(width: int, threshold: int = COLLAPSE_BELOW) -> bool:
    """True when ``width`` is too narrow for a side-by-side split."""
    return width < threshold


def truncate(line: str, width: int) -> str:
    """Clip a styled line to ``width`` visible cells, resetting styles."""
    if visible_length(line) <= width:
        return line
    from termflow.ansi.utils import ANSI_ESCAPE_RE

    out: list[str] = []
    used = 0
    i = 0
    while i < len(line) and used < width - 1:
        m = ANSI_ESCAPE_RE.match(line, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        out.append(line[i])
        used += visible_length(line[i])
        i += 1
    return "".join(out) + f"{RESET}\u2026"


def two_columns(left: list[str], right: list[str], left_width: int, total_width: int) -> list[str]:
    """Join two line-lists side by side, padding the left column."""
    divider = f" {DIM_ON}\u2502{RESET} "
    right_width = max(0, total_width - left_width - 3)
    merged: list[str] = []
    for i in range(max(len(left), len(right))):
        lline = left[i] if i < len(left) else ""
        rline = truncate(right[i], right_width) if i < len(right) else ""
        pad = " " * max(0, left_width - visible_length(lline))
        merged.append(f"{lline}{pad}{divider}{rline}")
    return merged


def split_frame(
    left: list[str],
    right: list[str],
    *,
    width: int,
    list_width: int,
    focus: Literal["left", "right"] = "left",
    threshold: int = COLLAPSE_BELOW,
) -> list[str]:
    """Compose a master-detail body responsively.

    Wide terminals get both panes side by side (``left`` padded to
    ``list_width``, a dim divider, ``right`` truncated to the rest).
    Narrow terminals get ONLY the ``focus`` pane at full width -- the
    two-phase flow: pick in the master pane, then the detail pane takes
    over the whole screen.
    """
    if collapsed(width, threshold):
        pane = left if focus == "left" else right
        return [truncate(line, width) for line in pane]
    return two_columns(left, right, list_width, width)
