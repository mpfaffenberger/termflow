"""Rendering module for terminal output.

This module provides the complete rendering pipeline for converting
markdown parse events into beautiful ANSI-formatted terminal output.

Main components:
- Renderer: Main rendering class that handles all event types
- RenderStyle: Color and style configuration
- RenderFeatures: Feature flags for rendering behavior
- Specialized renderers for headings, code, lists, tables

Example:
    >>> from termflow.render import Renderer, RenderStyle, render_markdown

    >>> # Quick one-liner
    >>> render_markdown("# Hello\\n\\nThis is **bold**!")

    >>> # Custom styled renderer
    >>> style = RenderStyle.dracula()
    >>> renderer = Renderer(width=80, style=style)
    >>> events = parser.parse_document(markdown)
    >>> renderer.render_all(events)
"""

from termflow.render.code import (
    CODEPAD_HORIZ,
    render_code_block,
    render_code_end,
    render_code_line,
    render_code_start,
)
from termflow.render.heading import render_heading, render_heading_simple
from termflow.render.list import (
    BULLETS,
    get_bullet,
    get_ordered_bullet,
    render_list_continuation,
    render_list_item,
    to_roman,
)
from termflow.render.renderer import (
    Renderer,
    render_markdown,
    render_streaming,
)
from termflow.render.style import RenderFeatures, RenderStyle
from termflow.render.table import (
    TABLE_BOTTOM_LEFT,
    TABLE_BOTTOM_RIGHT,
    TABLE_CROSS,
    TABLE_HORIZ,
    TABLE_T_DOWN,
    TABLE_T_LEFT,
    TABLE_T_RIGHT,
    TABLE_T_UP,
    TABLE_TOP_LEFT,
    TABLE_TOP_RIGHT,
    TABLE_VERT,
    TableRenderState,
    render_table_bottom,
    render_table_complete,
    render_table_row,
    render_table_separator,
    render_table_top,
)
from termflow.render.text import (
    center,
    pad_left,
    pad_right,
    simple_wrap,
    text_wrap,
    truncate,
)

__all__ = [
    # List
    "BULLETS",
    # Code
    "CODEPAD_HORIZ",
    # Table
    "TABLE_BOTTOM_LEFT",
    "TABLE_BOTTOM_RIGHT",
    "TABLE_CROSS",
    "TABLE_HORIZ",
    "TABLE_TOP_LEFT",
    "TABLE_TOP_RIGHT",
    "TABLE_T_DOWN",
    "TABLE_T_LEFT",
    "TABLE_T_RIGHT",
    "TABLE_T_UP",
    "TABLE_VERT",
    # Style
    "RenderFeatures",
    "RenderStyle",
    # Main renderer
    "Renderer",
    "TableRenderState",
    # Text
    "center",
    "get_bullet",
    "get_ordered_bullet",
    "pad_left",
    "pad_right",
    "render_code_block",
    "render_code_end",
    "render_code_line",
    "render_code_start",
    # Heading
    "render_heading",
    "render_heading_simple",
    "render_list_continuation",
    "render_list_item",
    "render_markdown",
    "render_streaming",
    "render_table_bottom",
    "render_table_complete",
    "render_table_row",
    "render_table_separator",
    "render_table_top",
    "simple_wrap",
    "text_wrap",
    "to_roman",
    "truncate",
]
