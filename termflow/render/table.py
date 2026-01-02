"""Table rendering with Unicode borders.

Renders markdown tables with:
- Unicode box drawing borders
- Column alignment support
- Header row styling
- Automatic column width calculation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from termflow.ansi import BOLD_OFF, BOLD_ON, RESET, fg_color, visible_length

if TYPE_CHECKING:
    from termflow.render.style import RenderStyle

# =============================================================================
# Box Drawing Characters
# =============================================================================

TABLE_TOP_LEFT = "┌"
TABLE_TOP_RIGHT = "┐"
TABLE_BOTTOM_LEFT = "└"
TABLE_BOTTOM_RIGHT = "┘"
TABLE_HORIZ = "─"
TABLE_VERT = "│"
TABLE_CROSS = "┼"
TABLE_T_DOWN = "┬"
TABLE_T_UP = "┴"
TABLE_T_RIGHT = "├"
TABLE_T_LEFT = "┤"


@dataclass
class TableRenderState:
    """Track table rendering state across rows."""

    column_widths: list[int] = field(default_factory=list)
    alignments: list[str] = field(default_factory=list)  # 'left', 'center', 'right', 'none'
    is_header_done: bool = False
    row_count: int = 0

    def reset(self) -> None:
        """Reset state for a new table."""
        self.column_widths.clear()
        self.alignments.clear()
        self.is_header_done = False
        self.row_count = 0

    def update_widths(self, cells: list[str] | tuple[str, ...]) -> None:
        """Update column widths based on cell contents."""
        for i, cell in enumerate(cells):
            width = visible_length(str(cell).strip())
            if i >= len(self.column_widths):
                self.column_widths.append(width)
            else:
                self.column_widths[i] = max(self.column_widths[i], width)

    def set_alignments(self, alignments: list[str] | tuple[str, ...]) -> None:
        """Set column alignments."""
        self.alignments = list(alignments)

    def get_alignment(self, index: int) -> str:
        """Get alignment for a column."""
        if index < len(self.alignments):
            return self.alignments[index]
        return "left"

    def end_header(self) -> None:
        """Mark header as complete."""
        self.is_header_done = True


def _align_cell(text: str, width: int, alignment: str) -> str:
    """Align cell content within width using visible length."""
    text = text.strip()
    vis_len = visible_length(text)
    pad = max(0, width - vis_len)

    if alignment == "center":
        left = pad // 2
        right = pad - left
        return " " * left + text + " " * right
    elif alignment == "right":
        return " " * pad + text
    else:  # left or none
        return text + " " * pad


def render_table_top(
    state: TableRenderState,
    margin: str,
    style: RenderStyle,
) -> str:
    """Render table top border."""
    fg = fg_color(style.symbol)

    parts = []
    for w in state.column_widths:
        parts.append(TABLE_HORIZ * (w + 2))  # +2 for cell padding

    border = TABLE_T_DOWN.join(parts)
    return f"{margin}{fg}{TABLE_TOP_LEFT}{border}{TABLE_TOP_RIGHT}{RESET}"


def render_table_row(
    cells: list[str] | tuple[str, ...],
    state: TableRenderState,
    _width: int,  # Reserved for future width constraints
    margin: str,
    style: RenderStyle,
    is_header: bool = False,
) -> list[str]:
    """Render a table row.

    Args:
        cells: Cell contents
        state: Table rendering state
        width: Available width
        margin: Left margin
        style: Render style
        is_header: Whether this is a header row

    Returns:
        Rendered lines.
    """
    # Update column widths
    state.update_widths(cells)

    fg = fg_color(style.symbol)
    lines: list[str] = []

    # Build the row content
    cell_parts = []
    for i, cell in enumerate(cells):
        col_width = state.column_widths[i] if i < len(state.column_widths) else len(str(cell))
        alignment = state.get_alignment(i)
        aligned = _align_cell(str(cell), col_width, alignment)

        if is_header:
            # Bold header cells
            cell_parts.append(f"{BOLD_ON}{aligned}{BOLD_OFF}")
        else:
            cell_parts.append(aligned)

    # Join cells with vertical bars
    row_content = f" {fg}{TABLE_VERT}{RESET} ".join(cell_parts)
    row = f"{margin}{fg}{TABLE_VERT}{RESET} {row_content} {fg}{TABLE_VERT}{RESET}"

    lines.append(row)
    state.row_count += 1

    return lines


def render_table_separator(
    state: TableRenderState,
    margin: str,
    style: RenderStyle,
) -> str:
    """Render a table separator row (between header and body)."""
    fg = fg_color(style.symbol)

    parts = []
    for w in state.column_widths:
        parts.append(TABLE_HORIZ * (w + 2))  # +2 for padding

    sep = TABLE_CROSS.join(parts)
    return f"{margin}{fg}{TABLE_T_RIGHT}{sep}{TABLE_T_LEFT}{RESET}"


def render_table_bottom(
    state: TableRenderState,
    margin: str,
    style: RenderStyle,
) -> str:
    """Render table bottom border."""
    fg = fg_color(style.symbol)

    parts = []
    for w in state.column_widths:
        parts.append(TABLE_HORIZ * (w + 2))

    border = TABLE_T_UP.join(parts)
    return f"{margin}{fg}{TABLE_BOTTOM_LEFT}{border}{TABLE_BOTTOM_RIGHT}{RESET}"


def render_table_complete(
    header: list[str] | tuple[str, ...],
    rows: list[list[str] | tuple[str, ...]],
    alignments: list[str],
    width: int,
    margin: str,
    style: RenderStyle,
) -> list[str]:
    """Render a complete table.

    Convenience function for rendering tables in one call.

    Args:
        header: Header row cells
        rows: Body row cells
        alignments: Column alignments
        width: Available width
        margin: Left margin
        style: Render style

    Returns:
        All rendered lines.
    """
    state = TableRenderState()

    # First pass: calculate column widths
    state.update_widths(header)
    for row in rows:
        state.update_widths(row)

    state.set_alignments(alignments)

    # Render
    lines: list[str] = []

    lines.append(render_table_top(state, margin, style))
    lines.extend(render_table_row(header, state, width, margin, style, is_header=True))
    lines.append(render_table_separator(state, margin, style))
    state.end_header()

    for row in rows:
        lines.extend(render_table_row(row, state, width, margin, style, is_header=False))

    lines.append(render_table_bottom(state, margin, style))

    return lines
