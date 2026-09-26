"""Heading rendering for H1-H6.

Each heading level has a distinct visual style:
- H1: Bold, left-justified, bright color
- H2: Bold, bright color
- H3: Bold, head color
- H4: Bold, default color
- H5: Normal text
- H6: Dim grey

Long headings word-wrap to the available width; H1/H2 underlines span
the widest wrapped line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from termflow.ansi import BOLD_OFF, BOLD_ON, RESET, fg_color, visible_length
from termflow.render.text import text_wrap

if TYPE_CHECKING:
    from termflow.render.style import RenderStyle

#: Marker shown before H5 headings.
H5_MARKER = "▸"


def _heading_codes(level: int, style: RenderStyle) -> tuple[str, str]:
    """Return the (open, close) ANSI codes that style a heading's text."""
    if level <= 2:
        return f"{BOLD_ON}{fg_color(style.bright)}", RESET
    if level == 3:
        return f"{BOLD_ON}{fg_color(style.head)}", RESET
    if level == 4:
        return BOLD_ON, BOLD_OFF
    if level == 5:
        return "", ""
    return fg_color(style.grey), RESET


def render_heading(
    level: int,
    content: str,
    width: int,
    margin: str,
    style: RenderStyle,
) -> list[str]:
    """Render a heading (H1-H6), word-wrapped to ``width``.

    Args:
        level: Heading level (1-6)
        content: Heading text content
        width: Available width for rendering (excluding ``margin``)
        margin: Left margin string (for blockquotes, etc.)
        style: Render style configuration

    Returns:
        List of rendered lines: the (possibly wrapped) heading text, plus
        an underline for H1/H2.

    Example:
        >>> lines = render_heading(1, "Hello World", 80, "", style)
        >>> print(lines[0])  # Bold, colored
    """
    open_code, close_code = _heading_codes(level, style)
    if level == 5:
        first_prefix = f"{fg_color(style.symbol)}{H5_MARKER}{RESET} "
        cont_prefix = " " * (visible_length(H5_MARKER) + 1)
    else:
        first_prefix = cont_prefix = ""

    text_lines = text_wrap(f"{open_code}{content}{close_code}", width, 0, first_prefix, cont_prefix)
    lines = [f"{margin}{line}" for line in text_lines]

    if level <= 2:
        fg = fg_color(style.bright)
        text_width = max((visible_length(line) for line in text_lines), default=0)
        # H1 gets a wider decorative double rule, H2 a subtle single one.
        char, extra = ("═", 4) if level == 1 else ("─", 0)
        lines.append(f"{margin}{fg}{char * min(text_width + extra, width)}{RESET}")

    return lines


def render_heading_simple(
    level: int,
    content: str,
    style: RenderStyle,
) -> str:
    """Render a heading as a single line (no wrapping/decorations).

    Useful for inline contexts or simple output.

    Args:
        level: Heading level (1-6)
        content: Heading text
        style: Render style

    Returns:
        Formatted heading string.
    """
    open_code, close_code = _heading_codes(level, style)
    return f"{open_code}{content}{close_code}"
