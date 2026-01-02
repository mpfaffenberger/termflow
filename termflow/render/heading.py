"""Heading rendering for H1-H6.

Each heading level has a distinct visual style:
- H1: Bold, left-justified, bright color
- H2: Bold, bright color
- H3: Bold, head color
- H4: Bold, default color
- H5: Normal text
- H6: Dim grey
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from termflow.ansi import BOLD_OFF, BOLD_ON, RESET, fg_color, visible_length

if TYPE_CHECKING:
    from termflow.render.style import RenderStyle


def render_heading(
    level: int,
    content: str,
    width: int,
    margin: str,
    style: RenderStyle,
) -> list[str]:
    """Render a heading (H1-H6).

    Args:
        level: Heading level (1-6)
        content: Heading text content
        width: Available width for rendering
        margin: Left margin string (for blockquotes, etc.)
        style: Render style configuration

    Returns:
        List of rendered lines (usually just one).

    Example:
        >>> lines = render_heading(1, "Hello World", 80, "", style)
        >>> print(lines[0])  # Centered, bold, colored
    """
    lines: list[str] = []

    if level == 1:
        # H1: Bold, left-justified, bright color, with decorative underline
        fg = fg_color(style.bright)
        text = f"{BOLD_ON}{fg}{content}{RESET}"

        # Left-justify the heading
        lines.append(f"{margin}{text}")

        # Add decorative underline
        underline_char = "═"
        underline = underline_char * min(visible_length(content) + 4, width)
        lines.append(f"{margin}{fg}{underline}{RESET}")

    elif level == 2:
        # H2: Bold, bright color, with subtle underline
        fg = fg_color(style.bright)
        lines.append(f"{margin}{BOLD_ON}{fg}{content}{RESET}")

        # Subtle underline
        underline = "─" * min(visible_length(content), width)
        lines.append(f"{margin}{fg}{underline}{RESET}")

    elif level == 3:
        # H3: Bold, head color
        fg = fg_color(style.head)
        lines.append(f"{margin}{BOLD_ON}{fg}{content}{RESET}")

    elif level == 4:
        # H4: Just bold
        lines.append(f"{margin}{BOLD_ON}{content}{BOLD_OFF}")

    elif level == 5:
        # H5: Normal text, slightly emphasized with symbol color
        fg = fg_color(style.symbol)
        lines.append(f"{margin}{fg}▸{RESET} {content}")

    else:
        # H6: Dim grey
        fg = fg_color(style.grey)
        lines.append(f"{margin}{fg}{content}{RESET}")

    return lines


def render_heading_simple(
    level: int,
    content: str,
    style: RenderStyle,
) -> str:
    """Render a heading as a single line (no centering/decorations).

    Useful for inline contexts or simple output.

    Args:
        level: Heading level (1-6)
        content: Heading text
        style: Render style

    Returns:
        Formatted heading string.
    """
    if level <= 2:
        fg = fg_color(style.bright)
        return f"{BOLD_ON}{fg}{content}{RESET}"
    elif level == 3:
        fg = fg_color(style.head)
        return f"{BOLD_ON}{fg}{content}{RESET}"
    elif level == 4:
        return f"{BOLD_ON}{content}{BOLD_OFF}"
    elif level == 5:
        return content
    else:
        fg = fg_color(style.grey)
        return f"{fg}{content}{RESET}"
