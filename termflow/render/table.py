"""Table rendering with Unicode borders.

Renders markdown tables with:
- Unicode box drawing borders
- Column alignment support
- Header row styling
- Automatic column width calculation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from termflow.ansi import BOLD_OFF, BOLD_ON, RESET, fg_color, visible_length
from termflow.ansi.utils import wrap_ansi

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

    def cap_widths_to_max(self, margin_width: int, available_width: int | None = None) -> None:
        """Cap column widths so the rendered table fits horizontally.

        The effective cap is the minimum of ``available_width`` (the total
        horizontal space available for the rendered table, including outer
        margin) and ``TERMFLOW_MAX_TABLE_WIDTH`` (if set). Column widths are
        redistributed proportionally when the table would overflow. Content
        that no longer fits a cell will be wrapped at render time.
        """
        max_w: int | None = available_width
        env_max = os.environ.get("TERMFLOW_MAX_TABLE_WIDTH")
        if env_max:
            try:
                env_val = int(env_max)
                max_w = min(max_w, env_val) if max_w is not None else env_val
            except ValueError:
                pass

        if max_w is None:
            return

        num_cols = len(self.column_widths)
        if num_cols == 0:
            return

        # overhead: margin + outer borders (2) + cell padding (2 per col: " X ")
        # + inner borders (num_cols - 1)
        overhead = margin_width + 2 + (num_cols * 2) + (num_cols - 1)
        available = max_w - overhead

        if available <= num_cols:
            return  # Can't shrink further

        total_content = sum(self.column_widths)
        if total_content <= available:
            return  # Already fits

        # Proportionally redistribute
        min_col = 8  # Minimum column width
        new_widths = []
        for w in self.column_widths:
            ratio = w / total_content
            new_w = max(min_col, int(available * ratio))
            new_widths.append(new_w)

        # Adjust rounding errors
        diff = available - sum(new_widths)
        if diff > 0:
            # Give extra space to the widest column
            idx = new_widths.index(max(new_widths))
            new_widths[idx] += diff
        elif diff < 0:
            # Shrink the widest column
            idx = new_widths.index(max(new_widths))
            new_widths[idx] = max(min_col, new_widths[idx] + diff)

        self.column_widths = new_widths

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


def _wrap_cell(text: str, width: int) -> list[str]:
    """Wrap cell text to fit within width, returning lines."""
    text = text.strip()
    if visible_length(text) <= width:
        return [text]
    wrapped = wrap_ansi(text, width)
    return wrapped if wrapped else [text]


def render_table_row(
    cells: list[str] | tuple[str, ...],
    state: TableRenderState,
    _width: int,  # Reserved for future width constraints
    margin: str,
    style: RenderStyle,
    is_header: bool = False,
) -> list[str]:
    """Render a table row with text wrapping support.

    Args:
        cells: Cell contents
        state: Table rendering state
        _width: Available width
        margin: Left margin
        style: Render style
        is_header: Whether this is a header row

    Returns:
        Rendered lines.
    """
    fg = fg_color(style.symbol)
    lines: list[str] = []

    # Wrap each cell's content to its column width
    wrapped_cells: list[list[str]] = []
    for i, cell in enumerate(cells):
        col_width = state.column_widths[i] if i < len(state.column_widths) else len(str(cell))
        wrapped_cells.append(_wrap_cell(str(cell), col_width))

    # Number of visual lines this row needs
    max_lines = max((len(wc) for wc in wrapped_cells), default=1)

    for line_idx in range(max_lines):
        cell_parts = []
        for i, wc in enumerate(wrapped_cells):
            col_width = state.column_widths[i] if i < len(state.column_widths) else 10
            alignment = state.get_alignment(i)
            text = wc[line_idx] if line_idx < len(wc) else ""
            aligned = _align_cell(text, col_width, alignment)

            if is_header:
                cell_parts.append(f"{BOLD_ON}{aligned}{BOLD_OFF}")
            else:
                cell_parts.append(aligned)

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

    # Cap column widths so borders don't wrap past the terminal edge.
    # ``width`` is the content budget from the renderer (already minus margin),
    # so total available = width + margin_width.
    margin_width = visible_length(margin)
    state.cap_widths_to_max(margin_width, available_width=width + margin_width)

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
