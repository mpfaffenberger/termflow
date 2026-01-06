"""Mermaid diagram parsing, layout, and rendering.

This module provides in-core Mermaid flowchart parsing, layout computation,
image rendering, and terminal graphics output for termflow. When parsing
fails, a MermaidParseError is raised so the renderer can fall back to
styled code display.

Supported Mermaid features:
- Flowchart/graph with directions (LR, RL, TB, TD, BT)
- Node shapes: rectangle, rounded, diamond, circle, flag
- Edge styles: solid, dotted, thick (with or without arrows)
- Edge labels

Terminal graphics protocols:
- Kitty Graphics Protocol (for Kitty terminal)
- iTerm2 Inline Images (for iTerm2/WezTerm)
- Unicode Block Characters (universal fallback)

Example:
    >>> from termflow.render.mermaid import render_mermaid_to_terminal
    >>>
    >>> output = render_mermaid_to_terminal('''
    ...     graph LR
    ...         A[Start] --> B{Decision}
    ...         B -->|Yes| C[End]
    ... ''')
    >>> print(output)
"""

from termflow.render.mermaid.canvas import (
    RenderConfig,
    render_to_bytes,
    render_to_image,
)
from termflow.render.mermaid.graphics import (
    GraphicsProtocol,
    detect_graphics_protocol,
    image_to_blocks,
    image_to_iterm2,
    image_to_kitty,
    image_to_terminal,
    is_graphics_supported,
    render_mermaid_to_terminal,
)
from termflow.render.mermaid.layout import (
    GraphLayout,
    PositionedEdge,
    PositionedNode,
    layout_graph,
)
from termflow.render.mermaid.parser import (
    Direction,
    Edge,
    EdgeStyle,
    MermaidGraph,
    MermaidParseError,
    MermaidParser,
    Node,
    NodeShape,
    is_mermaid_flowchart,
    parse_mermaid,
)

__all__ = [
    # Enums
    "Direction",
    "EdgeStyle",
    "GraphicsProtocol",
    "NodeShape",
    # Parser data classes
    "Edge",
    "MermaidGraph",
    "Node",
    # Layout data classes
    "GraphLayout",
    "PositionedEdge",
    "PositionedNode",
    # Render config
    "RenderConfig",
    # Parser
    "MermaidParser",
    "MermaidParseError",
    # Functions
    "detect_graphics_protocol",
    "image_to_blocks",
    "image_to_iterm2",
    "image_to_kitty",
    "image_to_terminal",
    "is_graphics_supported",
    "is_mermaid_flowchart",
    "layout_graph",
    "parse_mermaid",
    "render_mermaid_to_terminal",
    "render_to_bytes",
    "render_to_image",
]
