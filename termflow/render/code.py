"""Code block rendering with simple horizontal separators.

Renders code blocks with:
- Simple horizontal line separators (top/bottom)
- Language label in the top border
- Background coloring
- Syntax highlighting integration

Note: Box corners and side borders are intentionally omitted to make copying
code easier - users can select code without grabbing any border characters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from termflow.ansi import RESET, fg_color, visible_length, wrap_ansi

if TYPE_CHECKING:
    from termflow.render.style import RenderStyle

# =============================================================================
# Box Drawing Characters
# =============================================================================

CODEPAD_HORIZ = "─"

#: Marks a code line that was wrapped because it overflowed the width.
CODE_CONTINUATION = "↪ "


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
    inner_width = width  # full width, no corners

    if label_len > 0:
        # Put label on the left side
        right_border_len = inner_width - label_len
        right_border = CODEPAD_HORIZ * max(0, right_border_len)
        top_line = f"{margin}{grey}{lang_label}{fg}{right_border}{RESET}"
    else:
        border = CODEPAD_HORIZ * inner_width
        top_line = f"{margin}{fg}{border}{RESET}"

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
    fg_color(style.symbol)

    # Calculate visible length and padding
    vis_len = visible_length(highlighted)

    if pretty_pad:
        # No side borders for clean copy-paste! Just the code.
        content_width = width
        padding = max(0, content_width - vis_len)
        return f"{margin}{highlighted}{' ' * padding}"
    else:
        padding = max(0, width - vis_len)
        return f"{margin}{highlighted}{' ' * padding}"


def wrap_code_line(highlighted: str, width: int, style: RenderStyle) -> list[str]:
    """Hard-wrap an overflowing code line, marking continuations with ``↪``.

    Code is split at exactly the available width (whitespace is
    significant, so no word-boundary games), and syntax highlighting is
    carried across the break. Lines that fit are returned untouched.

    Args:
        highlighted: Syntax-highlighted line with ANSI codes
        width: Available width
        style: Render style (for the continuation marker color)

    Returns:
        One or more display lines.
    """
    if visible_length(highlighted) <= width:
        return [highlighted]
    marker = f"{fg_color(style.grey)}{CODE_CONTINUATION}{RESET}"
    chunks = wrap_ansi(
        highlighted, max(1, width - visible_length(CODE_CONTINUATION)), break_words=True
    )
    return [chunks[0], *(f"{marker}{chunk}" for chunk in chunks[1:])]


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
    border = CODEPAD_HORIZ * width

    lines.append(f"{margin}{fg}{border}{RESET}")
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
        for chunk in wrap_code_line(highlighted, width, style):
            result.append(render_code_line(original, chunk, width, margin, style, pretty_pad))

    # End
    result.extend(render_code_end(width, margin, style, pretty_pad))

    return result
