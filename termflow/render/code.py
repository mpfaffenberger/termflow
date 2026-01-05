"""Code block rendering with box drawing borders.

Renders code blocks with:
- Unicode box drawing borders (top/bottom frames, right-side border only)
- Language label in the top border
- Background coloring
- Syntax highlighting integration

Note: Left-side vertical borders are intentionally omitted to make copying
code easier - users can select from the left edge without grabbing border chars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from termflow.ansi import RESET, fg_color, visible_length

if TYPE_CHECKING:
    from termflow.render.style import RenderStyle

# =============================================================================
# Box Drawing Characters
# =============================================================================

CODEPAD_TOP_LEFT = "╭"
CODEPAD_TOP_RIGHT = "╮"
CODEPAD_BOTTOM_LEFT = "╰"
CODEPAD_BOTTOM_RIGHT = "╯"
CODEPAD_HORIZ = "─"
CODEPAD_VERT = "│"


def render_code_start(
    language: str | None,
    width: int,
    margin: str,
    style: RenderStyle,
    pretty_pad: bool = True,
) -> list[str]:
    """Render the start of a code block.

    Args:
        language: Programming language (for label)
        width: Available width
        margin: Left margin string
        style: Render style
        pretty_pad: Whether to add decorative borders

    Returns:
        Lines for the code block header.
    """
    lines: list[str] = []

    if not pretty_pad:
        return lines

    fg = fg_color(style.symbol)
    grey = fg_color(style.grey)

    # Language label
    lang_label = f" {language} " if language else ""

    # Calculate border lengths
    label_len = len(lang_label)
    inner_width = width - 2  # -2 for corners

    if label_len > 0:
        # Put label on the right side
        left_border_len = inner_width - label_len
        left_border = CODEPAD_HORIZ * max(0, left_border_len)
        top_line = (
            f"{margin}{fg}{CODEPAD_TOP_LEFT}{left_border}"
            f"{grey}{lang_label}{fg}{CODEPAD_TOP_RIGHT}{RESET}"
        )
    else:
        border = CODEPAD_HORIZ * inner_width
        top_line = f"{margin}{fg}{CODEPAD_TOP_LEFT}{border}{CODEPAD_TOP_RIGHT}{RESET}"

    lines.append(top_line)
    return lines


def render_code_line(
    _line: str,  # Original line for width calculation (unused currently)
    highlighted: str,
    width: int,
    margin: str,
    style: RenderStyle,
    pretty_pad: bool = True,
) -> str:
    """Render a single line of code.

    Args:
        line: Original line (for width calculation)
        highlighted: Syntax-highlighted version with ANSI codes
        width: Available width
        margin: Left margin string
        style: Render style
        pretty_pad: Whether to add side borders

    Returns:
        Formatted code line.
    """
    fg = fg_color(style.symbol)

    # Calculate visible length and padding
    vis_len = visible_length(highlighted)

    if pretty_pad:
        # No left border for easier copying! Just indent to align with box corners.
        # Right border closes the visual box.
        content_width = width - 4  # space + content + space + right border
        padding = max(0, content_width - vis_len)
        return f"{margin}  {highlighted}{' ' * padding} {fg}{CODEPAD_VERT}{RESET}"
    else:
        padding = max(0, width - vis_len)
        return f"{margin}{highlighted}{' ' * padding}"


def render_code_end(
    width: int,
    margin: str,
    style: RenderStyle,
    pretty_pad: bool = True,
) -> list[str]:
    """Render the end of a code block.

    Args:
        width: Available width
        margin: Left margin string
        style: Render style
        pretty_pad: Whether to add decorative borders

    Returns:
        Lines for the code block footer.
    """
    lines: list[str] = []

    if not pretty_pad:
        return lines

    fg = fg_color(style.symbol)
    inner_width = width - 2
    border = CODEPAD_HORIZ * inner_width

    lines.append(f"{margin}{fg}{CODEPAD_BOTTOM_LEFT}{border}{CODEPAD_BOTTOM_RIGHT}{RESET}")
    return lines


def render_code_block(
    code: str,
    language: str | None,
    highlighted_lines: list[str],
    width: int,
    margin: str,
    style: RenderStyle,
    pretty_pad: bool = True,
) -> list[str]:
    """Render a complete code block.

    Convenience function that combines start, lines, and end.

    Args:
        code: Original code (for width calculation)
        language: Programming language
        highlighted_lines: Pre-highlighted lines
        width: Available width
        margin: Left margin
        style: Render style
        pretty_pad: Whether to add borders

    Returns:
        All rendered lines.
    """
    result: list[str] = []

    # Start
    result.extend(render_code_start(language, width, margin, style, pretty_pad))

    # Lines
    original_lines = code.splitlines()
    for i, highlighted in enumerate(highlighted_lines):
        original = original_lines[i] if i < len(original_lines) else ""
        result.append(render_code_line(original, highlighted, width, margin, style, pretty_pad))

    # End
    result.extend(render_code_end(width, margin, style, pretty_pad))

    return result
