"""Canvas rendering for Mermaid diagrams using Pillow.

Renders positioned graph layouts to PNG images with proper styling
for nodes, edges, and labels.

Example:
    >>> from termflow.render.mermaid import parse_mermaid, layout_graph
    >>> from termflow.render.mermaid.canvas import render_to_image
    >>>
    >>> graph = parse_mermaid('''
    ...     graph LR
    ...         A[Start] --> B[End]
    ... ''')
    >>> layout = layout_graph(graph)
    >>> image = render_to_image(layout)
    >>> image.save("diagram.png")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from termflow.render.mermaid.layout import GraphLayout, PositionedEdge, PositionedNode
from termflow.render.mermaid.parser import Direction, EdgeStyle, NodeShape

if TYPE_CHECKING:
    pass


# =============================================================================
# Type Aliases
# =============================================================================

RGBA = tuple[int, int, int, int]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RenderConfig:
    """Configuration for diagram rendering.

    All colors are RGBA tuples (red, green, blue, alpha) where each
    component is 0-255.

    Attributes:
        background_color: Background color (default: transparent)
        node_fill: Node fill color (default: white)
        node_stroke: Node border color (default: black)
        text_color: Text color (default: black)
        edge_color: Edge/arrow color (default: black)
        stroke_width: Line width for borders and edges
        font_size: Font size for labels
        padding: Padding around the entire diagram
        scale: Scale factor for character units to pixels
        size_multiplier: Global multiplier for all sizes (for bigger diagrams)
    """

    background_color: RGBA = (255, 255, 255, 0)  # Transparent
    node_fill: RGBA = (255, 255, 255, 255)  # White
    node_stroke: RGBA = (0, 0, 0, 255)  # Black
    text_color: RGBA = (0, 0, 0, 255)  # Black
    edge_color: RGBA = (100, 100, 100, 255)  # Dark gray
    label_bg_color: RGBA = (255, 255, 255, 230)  # Semi-transparent white
    stroke_width: int = 4  # Thicker lines for visibility
    font_size: int = 28  # Bigger, bolder text
    padding: int = 60  # More breathing room
    scale: int = 14  # Larger pixels per character unit
    size_multiplier: float = 1.0  # Global size multiplier

    def __post_init__(self) -> None:
        """Apply size multiplier to all dimensional values."""
        if self.size_multiplier != 1.0:
            self.stroke_width = max(1, int(self.stroke_width * self.size_multiplier))
            self.font_size = max(8, int(self.font_size * self.size_multiplier))
            self.padding = int(self.padding * self.size_multiplier)
            self.scale = max(5, int(self.scale * self.size_multiplier))


# =============================================================================
# Font Loading
# =============================================================================

# Common system font paths to try
SYSTEM_FONTS = [
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/Library/Fonts/Arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font, trying system fonts first then falling back to default.

    Args:
        size: Desired font size

    Returns:
        Loaded font object
    """
    # Try system fonts
    for font_path in SYSTEM_FONTS:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except (OSError, IOError):
                continue

    # Fallback to default bitmap font
    try:
        return ImageFont.load_default(size)
    except TypeError:
        # Older Pillow versions don't support size argument
        return ImageFont.load_default()


# =============================================================================
# Drawing Helpers
# =============================================================================


def _draw_rectangle(
    draw: ImageDraw.ImageDraw,
    node: PositionedNode,
    config: RenderConfig,
    scale: int,
    offset_x: float,
    offset_y: float,
) -> None:
    """Draw a rectangle node."""
    x = (node.x + offset_x) * scale
    y = (node.y + offset_y) * scale
    w = node.width * scale
    h = node.height * scale

    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2

    draw.rectangle(
        [left, top, right, bottom],
        fill=config.node_fill,
        outline=config.node_stroke,
        width=config.stroke_width,
    )


def _draw_rounded(
    draw: ImageDraw.ImageDraw,
    node: PositionedNode,
    config: RenderConfig,
    scale: int,
    offset_x: float,
    offset_y: float,
) -> None:
    """Draw a rounded rectangle node."""
    x = (node.x + offset_x) * scale
    y = (node.y + offset_y) * scale
    w = node.width * scale
    h = node.height * scale

    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2

    # Radius is proportional to height
    radius = min(h / 3, w / 4, 15)

    draw.rounded_rectangle(
        [left, top, right, bottom],
        radius=radius,
        fill=config.node_fill,
        outline=config.node_stroke,
        width=config.stroke_width,
    )


def _draw_diamond(
    draw: ImageDraw.ImageDraw,
    node: PositionedNode,
    config: RenderConfig,
    scale: int,
    offset_x: float,
    offset_y: float,
) -> None:
    """Draw a diamond node."""
    x = (node.x + offset_x) * scale
    y = (node.y + offset_y) * scale
    w = node.width * scale
    h = node.height * scale

    # Diamond points: top, right, bottom, left
    points = [
        (x, y - h / 2),  # Top
        (x + w / 2, y),  # Right
        (x, y + h / 2),  # Bottom
        (x - w / 2, y),  # Left
    ]

    draw.polygon(
        points,
        fill=config.node_fill,
        outline=config.node_stroke,
        width=config.stroke_width,
    )


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    node: PositionedNode,
    config: RenderConfig,
    scale: int,
    offset_x: float,
    offset_y: float,
) -> None:
    """Draw a circle node."""
    x = (node.x + offset_x) * scale
    y = (node.y + offset_y) * scale
    # Use the smaller dimension to ensure a proper circle
    diameter = min(node.width, node.height) * scale

    left = x - diameter / 2
    top = y - diameter / 2
    right = x + diameter / 2
    bottom = y + diameter / 2

    draw.ellipse(
        [left, top, right, bottom],
        fill=config.node_fill,
        outline=config.node_stroke,
        width=config.stroke_width,
    )


def _draw_flag(
    draw: ImageDraw.ImageDraw,
    node: PositionedNode,
    config: RenderConfig,
    scale: int,
    offset_x: float,
    offset_y: float,
) -> None:
    """Draw a flag/asymmetric node (pointed on left side)."""
    x = (node.x + offset_x) * scale
    y = (node.y + offset_y) * scale
    w = node.width * scale
    h = node.height * scale

    # Flag shape: pointed on left, flat on right
    point_indent = w * 0.15  # How far the point goes in

    points = [
        (x - w / 2 + point_indent, y - h / 2),  # Top left (indented)
        (x + w / 2, y - h / 2),  # Top right
        (x + w / 2, y + h / 2),  # Bottom right
        (x - w / 2 + point_indent, y + h / 2),  # Bottom left (indented)
        (x - w / 2, y),  # Left point
    ]

    draw.polygon(
        points,
        fill=config.node_fill,
        outline=config.node_stroke,
        width=config.stroke_width,
    )


def _draw_node(
    draw: ImageDraw.ImageDraw,
    node: PositionedNode,
    config: RenderConfig,
    scale: int,
    offset_x: float,
    offset_y: float,
) -> None:
    """Draw a node based on its shape."""
    shape_drawers = {
        NodeShape.RECT: _draw_rectangle,
        NodeShape.ROUNDED: _draw_rounded,
        NodeShape.DIAMOND: _draw_diamond,
        NodeShape.CIRCLE: _draw_circle,
        NodeShape.FLAG: _draw_flag,
    }

    drawer = shape_drawers.get(node.shape, _draw_rectangle)
    drawer(draw, node, config, scale, offset_x, offset_y)


def _draw_node_label(
    draw: ImageDraw.ImageDraw,
    node: PositionedNode,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    config: RenderConfig,
    scale: int,
    offset_x: float,
    offset_y: float,
) -> None:
    """Draw centered text label inside a node."""
    x = (node.x + offset_x) * scale
    y = (node.y + offset_y) * scale

    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), node.label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center text in node
    text_x = x - text_width / 2
    text_y = y - text_height / 2

    draw.text((text_x, text_y), node.label, fill=config.text_color, font=font)


def _draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    angle: float,
    config: RenderConfig,
    size: float = 10,
) -> None:
    """Draw an arrow head pointing in the given direction.

    Args:
        draw: ImageDraw object
        x: X coordinate of arrow tip
        y: Y coordinate of arrow tip
        angle: Angle in radians (direction arrow points)
        config: Render configuration
        size: Size of arrow head
    """
    # Arrow head is a triangle
    # Calculate the two back points of the arrow
    back_angle = math.pi / 6  # 30 degrees spread

    # Left point of arrow
    left_x = x - size * math.cos(angle - back_angle)
    left_y = y - size * math.sin(angle - back_angle)

    # Right point of arrow
    right_x = x - size * math.cos(angle + back_angle)
    right_y = y - size * math.sin(angle + back_angle)

    points = [(x, y), (left_x, left_y), (right_x, right_y)]
    draw.polygon(points, fill=config.edge_color)


def _draw_dotted_line(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    config: RenderConfig,
    dash_length: int = 6,
    gap_length: int = 4,
) -> None:
    """Draw a dotted line from (x1, y1) to (x2, y2)."""
    # Calculate line length and direction
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)

    if length == 0:
        return

    # Normalize direction
    dx /= length
    dy /= length

    # Draw dashes
    segment_length = dash_length + gap_length
    pos = 0.0

    while pos < length:
        # Start of dash
        start_x = x1 + dx * pos
        start_y = y1 + dy * pos

        # End of dash (don't exceed total length)
        end_pos = min(pos + dash_length, length)
        end_x = x1 + dx * end_pos
        end_y = y1 + dy * end_pos

        draw.line(
            [(start_x, start_y), (end_x, end_y)],
            fill=config.edge_color,
            width=config.stroke_width,
        )

        pos += segment_length


def _draw_edge(
    draw: ImageDraw.ImageDraw,
    edge: PositionedEdge,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    config: RenderConfig,
    scale: int,
    offset_x: float,
    offset_y: float,
) -> None:
    """Draw an edge with optional arrow and label."""
    # Convert to pixel coordinates
    x1 = (edge.src_x + offset_x) * scale
    y1 = (edge.src_y + offset_y) * scale
    x2 = (edge.dst_x + offset_x) * scale
    y2 = (edge.dst_y + offset_y) * scale

    # Calculate angle for arrow head
    angle = math.atan2(y2 - y1, x2 - x1)

    # Shorten the line slightly if we have an arrow
    # Arrow size scales with stroke width for consistent look
    arrow_size = (config.stroke_width * 4) if edge.has_arrow else 0
    if edge.has_arrow:
        x2 -= arrow_size * 0.5 * math.cos(angle)
        y2 -= arrow_size * 0.5 * math.sin(angle)

    # Draw the line based on style
    if edge.style == EdgeStyle.DOTTED:
        _draw_dotted_line(draw, x1, y1, x2, y2, config)
    elif edge.style == EdgeStyle.THICK:
        draw.line(
            [(x1, y1), (x2, y2)],
            fill=config.edge_color,
            width=config.stroke_width * 2,
        )
    else:  # SOLID
        draw.line(
            [(x1, y1), (x2, y2)],
            fill=config.edge_color,
            width=config.stroke_width,
        )

    # Draw arrow head
    if edge.has_arrow:
        # Recalculate end point for arrow
        arrow_x = (edge.dst_x + offset_x) * scale
        arrow_y = (edge.dst_y + offset_y) * scale
        _draw_arrow_head(draw, arrow_x, arrow_y, angle, config, arrow_size)

    # Draw edge label if present
    if edge.label:
        # Center label on edge
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Get text dimensions
        bbox = draw.textbbox((0, 0), edge.label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw background for readability - scale with font size
        padding = max(4, config.font_size // 4)
        bg_left = mid_x - text_width / 2 - padding
        bg_top = mid_y - text_height / 2 - padding
        bg_right = mid_x + text_width / 2 + padding
        bg_bottom = mid_y + text_height / 2 + padding

        draw.rounded_rectangle(
            [bg_left, bg_top, bg_right, bg_bottom],
            radius=max(4, config.stroke_width * 2),
            fill=config.label_bg_color,
        )

        # Draw label text
        text_x = mid_x - text_width / 2
        text_y = mid_y - text_height / 2
        draw.text((text_x, text_y), edge.label, fill=config.text_color, font=font)


# =============================================================================
# Main Render Function
# =============================================================================


def render_to_image(
    layout: GraphLayout,
    config: RenderConfig | None = None,
) -> Image.Image:
    """Render a graph layout to a PIL Image.

    Args:
        layout: Positioned graph layout from layout_graph()
        config: Rendering configuration (uses defaults if None)

    Returns:
        PIL Image with the rendered diagram
    """
    if config is None:
        config = RenderConfig()

    # Handle empty layout
    if not layout.nodes:
        # Return a minimal image
        img = Image.new("RGBA", (100, 100), config.background_color)
        return img

    scale = config.scale
    padding = config.padding

    # Calculate image dimensions
    img_width = int(layout.width * scale + padding * 2)
    img_height = int(layout.height * scale + padding * 2)

    # Ensure minimum size
    img_width = max(img_width, 100)
    img_height = max(img_height, 100)

    # Create image with transparent/configured background
    img = Image.new("RGBA", (img_width, img_height), config.background_color)
    draw = ImageDraw.Draw(img)

    # Load font
    font = _load_font(config.font_size)

    # Calculate offset to center the diagram with padding
    offset_x = padding / scale
    offset_y = padding / scale

    # Draw edges first (so they appear behind nodes)
    for edge in layout.edges:
        _draw_edge(draw, edge, font, config, scale, offset_x, offset_y)

    # Draw nodes
    for node in layout.nodes.values():
        _draw_node(draw, node, config, scale, offset_x, offset_y)

    # Draw node labels (on top of nodes)
    for node in layout.nodes.values():
        _draw_node_label(draw, node, font, config, scale, offset_x, offset_y)

    return img


def render_to_bytes(
    layout: GraphLayout,
    config: RenderConfig | None = None,
    format: str = "PNG",
) -> bytes:
    """Render a graph layout to image bytes.

    Convenience function that renders to bytes directly.

    Args:
        layout: Positioned graph layout
        config: Rendering configuration
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Image data as bytes
    """
    import io

    img = render_to_image(layout, config)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()
