"""Graph layout engine using grandalf's Sugiyama algorithm.

Computes hierarchical layouts for Mermaid flowcharts, handling different
flow directions and producing positioned nodes and edges ready for rendering.

Example:
    >>> from termflow.render.mermaid import parse_mermaid
    >>> from termflow.render.mermaid.layout import layout_graph
    >>>
    >>> graph = parse_mermaid('''
    ...     graph LR
    ...         A[Start] --> B[End]
    ... ''')
    >>> layout = layout_graph(graph)
    >>> for node in layout.nodes.values():
    ...     print(f"{node.id}: ({node.x:.0f}, {node.y:.0f})")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from grandalf.graphs import Edge as GEdge
from grandalf.graphs import Graph as GGraph
from grandalf.graphs import Vertex as GVertex
from grandalf.layouts import SugiyamaLayout

from termflow.render.mermaid.parser import (
    Direction,
    EdgeStyle,
    MermaidGraph,
    NodeShape,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Configuration
# =============================================================================

# Default spacing between nodes - BIGGER for impressive diagrams!
DEFAULT_NODE_WIDTH = 18  # Characters wide (was 12)
DEFAULT_NODE_HEIGHT = 5  # Characters tall (was 3)
NODE_H_SPACING = 6  # Horizontal space between nodes (was 4)
NODE_V_SPACING = 4  # Vertical space between layers (was 2)

# Minimum padding around labels - more breathing room
LABEL_H_PADDING = 6  # 3 chars on each side (was 4)
LABEL_V_PADDING = 4  # 2 lines above and below (was 2)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PositionedNode:
    """A node with computed layout position.

    Coordinates represent the center of the node in character units.

    Attributes:
        id: Node identifier
        label: Display text
        shape: Visual shape (rect, diamond, etc.)
        x: X coordinate (center)
        y: Y coordinate (center)
        width: Node width in characters
        height: Node height in lines
    """

    id: str
    label: str
    shape: NodeShape
    x: float
    y: float
    width: float
    height: float


@dataclass
class PositionedEdge:
    """An edge with computed endpoint positions.

    Coordinates point to the edge of the source/destination nodes.

    Attributes:
        src: Source node ID
        dst: Destination node ID
        label: Optional edge label text
        style: Line style (solid, dotted, thick)
        has_arrow: Whether edge has an arrowhead
        src_x: X coordinate of edge start
        src_y: Y coordinate of edge start
        dst_x: X coordinate of edge end
        dst_y: Y coordinate of edge end
    """

    src: str
    dst: str
    label: str | None
    style: EdgeStyle
    has_arrow: bool
    src_x: float
    src_y: float
    dst_x: float
    dst_y: float


@dataclass
class GraphLayout:
    """Complete layout result for a graph.

    Attributes:
        nodes: Dictionary of positioned nodes by ID
        edges: List of positioned edges
        width: Total graph width in characters
        height: Total graph height in lines
        direction: Original flow direction
    """

    nodes: dict[str, PositionedNode] = field(default_factory=dict)
    edges: list[PositionedEdge] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    direction: Direction = Direction.TB


# =============================================================================
# Grandalf View Helper
# =============================================================================


class NodeView:
    """View class for grandalf layout integration.

    Grandalf expects vertices to have a `view` object with:
    - w: width
    - h: height
    - xy: tuple of (x, y) coordinates (updated by layout)
    """

    def __init__(self, width: float, height: float) -> None:
        self.w = width
        self.h = height
        self.xy = (0.0, 0.0)


# =============================================================================
# Layout Functions
# =============================================================================


def _compute_node_size(label: str, shape: NodeShape) -> tuple[float, float]:
    """Compute node dimensions based on label and shape.

    Args:
        label: Node label text
        shape: Node shape type

    Returns:
        Tuple of (width, height) in character units
    """
    # Base size from label length - with generous multiplier for readability
    label_width = len(label) * 1.5  # 1.5x multiplier for bigger text

    # Add padding based on shape - all shapes get more space!
    if shape == NodeShape.DIAMOND:
        # Diamonds need more space due to shape
        width = max(DEFAULT_NODE_WIDTH, label_width + LABEL_H_PADDING + 6)
        height = DEFAULT_NODE_HEIGHT + 3  # Extra vertical space for diamond
    elif shape == NodeShape.CIRCLE:
        # Circles should be roughly square and BIG
        size = max(DEFAULT_NODE_WIDTH, label_width + LABEL_H_PADDING + 2)
        width = size
        height = max(DEFAULT_NODE_HEIGHT, 7)  # Bigger circles
    elif shape == NodeShape.FLAG:
        # Flags need extra width for the arrow shape
        width = max(DEFAULT_NODE_WIDTH, label_width + LABEL_H_PADDING + 4)
        height = DEFAULT_NODE_HEIGHT
    else:
        # Rectangle, rounded - generous sizing
        width = max(DEFAULT_NODE_WIDTH, label_width + LABEL_H_PADDING)
        height = DEFAULT_NODE_HEIGHT

    return float(width), float(height)


def _transform_coordinates(
    x: float,
    y: float,
    direction: Direction,
    max_x: float,
    max_y: float,
) -> tuple[float, float]:
    """Transform coordinates based on flow direction.

    Grandalf produces top-to-bottom layout by default.
    We transform to match the requested direction.

    Args:
        x: Original X coordinate
        y: Original Y coordinate
        direction: Desired flow direction
        max_x: Maximum X in original coordinates
        max_y: Maximum Y in original coordinates

    Returns:
        Transformed (x, y) coordinates
    """
    if direction in (Direction.TB, Direction.TD):
        # Top to bottom - use as-is
        return x, y
    elif direction == Direction.BT:
        # Bottom to top - flip Y
        return x, max_y - y
    elif direction == Direction.LR:
        # Left to right - swap X and Y
        return y, x
    elif direction == Direction.RL:
        # Right to left - swap X and Y, then flip X
        return max_y - y, x
    else:
        # Fallback
        return x, y


def _compute_edge_endpoints(
    src_node: PositionedNode,
    dst_node: PositionedNode,
    direction: Direction,
) -> tuple[float, float, float, float]:
    """Compute edge start and end points at node boundaries.

    Args:
        src_node: Source positioned node
        dst_node: Destination positioned node
        direction: Flow direction

    Returns:
        Tuple of (src_x, src_y, dst_x, dst_y)
    """
    # Determine which sides of nodes to connect based on direction
    if direction in (Direction.TB, Direction.TD):
        # Top to bottom: exit bottom of src, enter top of dst
        src_x = src_node.x
        src_y = src_node.y + src_node.height / 2
        dst_x = dst_node.x
        dst_y = dst_node.y - dst_node.height / 2
    elif direction == Direction.BT:
        # Bottom to top: exit top of src, enter bottom of dst
        src_x = src_node.x
        src_y = src_node.y - src_node.height / 2
        dst_x = dst_node.x
        dst_y = dst_node.y + dst_node.height / 2
    elif direction == Direction.LR:
        # Left to right: exit right of src, enter left of dst
        src_x = src_node.x + src_node.width / 2
        src_y = src_node.y
        dst_x = dst_node.x - dst_node.width / 2
        dst_y = dst_node.y
    elif direction == Direction.RL:
        # Right to left: exit left of src, enter right of dst
        src_x = src_node.x - src_node.width / 2
        src_y = src_node.y
        dst_x = dst_node.x + dst_node.width / 2
        dst_y = dst_node.y
    else:
        # Fallback - center to center
        src_x, src_y = src_node.x, src_node.y
        dst_x, dst_y = dst_node.x, dst_node.y

    return src_x, src_y, dst_x, dst_y


def layout_graph(
    graph: MermaidGraph,
    node_width: int | None = None,
    node_height: int | None = None,
) -> GraphLayout:
    """Compute layout for a Mermaid graph using grandalf's Sugiyama algorithm.

    Args:
        graph: Parsed MermaidGraph to layout
        node_width: Override default node width (optional)
        node_height: Override default node height (optional)

    Returns:
        GraphLayout with positioned nodes and edges
    """
    result = GraphLayout(direction=graph.direction)

    # Handle empty graph
    if not graph.nodes:
        return result

    # Create grandalf vertices with views
    vertices: dict[str, GVertex] = {}
    for node_id, node in graph.nodes.items():
        v = GVertex(node_id)

        # Compute size based on label and shape
        if node_width is not None and node_height is not None:
            width, height = float(node_width), float(node_height)
        else:
            width, height = _compute_node_size(node.label, node.shape)

        v.view = NodeView(width, height)
        vertices[node_id] = v

    # Create grandalf edges
    g_edges: list[GEdge] = []
    for edge in graph.edges:
        src_v = vertices.get(edge.src)
        dst_v = vertices.get(edge.dst)
        if src_v and dst_v and src_v != dst_v:  # Skip self-loops for layout
            g_edges.append(GEdge(src_v, dst_v))

    # Create grandalf graph
    g_graph = GGraph(list(vertices.values()), g_edges)

    # Layout each connected component
    component_offset_x = 0.0
    max_height = 0.0

    for component in g_graph.C:
        # Run Sugiyama layout on this component
        sug = SugiyamaLayout(component)
        sug.init_all()
        sug.draw()

        # Find bounds of this component
        comp_min_x = float("inf")
        comp_min_y = float("inf")
        comp_max_x = float("-inf")
        comp_max_y = float("-inf")

        for v in component.sV:
            if v.view:
                vx, vy = v.view.xy
                vw, vh = v.view.w, v.view.h
                comp_min_x = min(comp_min_x, vx - vw / 2)
                comp_max_x = max(comp_max_x, vx + vw / 2)
                comp_min_y = min(comp_min_y, vy - vh / 2)
                comp_max_y = max(comp_max_y, vy + vh / 2)

        # Handle single node or degenerate component
        if comp_min_x == float("inf"):
            continue

        comp_width = comp_max_x - comp_min_x
        comp_height = comp_max_y - comp_min_y

        # Normalize positions within component and add offset
        for v in component.sV:
            if v.view:
                vx, vy = v.view.xy
                # Normalize to start at 0
                vx = vx - comp_min_x + component_offset_x
                vy = vy - comp_min_y
                v.view.xy = (vx, vy)

        # Track max height and update offset for next component
        max_height = max(max_height, comp_height)
        component_offset_x += comp_width + NODE_H_SPACING * 2

    # Now compute final bounds for coordinate transformation
    all_max_x = 0.0
    all_max_y = 0.0
    for v in vertices.values():
        if v.view:
            vx, vy = v.view.xy
            all_max_x = max(all_max_x, vx + v.view.w / 2)
            all_max_y = max(all_max_y, vy + v.view.h / 2)

    # Transform and create positioned nodes
    for node_id, node in graph.nodes.items():
        v = vertices.get(node_id)
        if v and v.view:
            x, y = v.view.xy
            width, height = v.view.w, v.view.h

            # Transform coordinates based on direction
            tx, ty = _transform_coordinates(
                x, y, graph.direction, all_max_x, all_max_y
            )

            # For LR/RL, swap width and height perception for bounds
            # but keep actual node dimensions
            result.nodes[node_id] = PositionedNode(
                id=node_id,
                label=node.label,
                shape=node.shape,
                x=tx,
                y=ty,
                width=width,
                height=height,
            )

    # Compute positioned edges
    for edge in graph.edges:
        src_node = result.nodes.get(edge.src)
        dst_node = result.nodes.get(edge.dst)

        if src_node and dst_node:
            # Handle self-loops specially
            if edge.src == edge.dst:
                # Self-loop: place edge to the right of node
                src_x = src_node.x + src_node.width / 2
                src_y = src_node.y - src_node.height / 4
                dst_x = src_node.x + src_node.width / 2
                dst_y = src_node.y + src_node.height / 4
            else:
                src_x, src_y, dst_x, dst_y = _compute_edge_endpoints(
                    src_node, dst_node, graph.direction
                )

            result.edges.append(
                PositionedEdge(
                    src=edge.src,
                    dst=edge.dst,
                    label=edge.label,
                    style=edge.style,
                    has_arrow=edge.has_arrow,
                    src_x=src_x,
                    src_y=src_y,
                    dst_x=dst_x,
                    dst_y=dst_y,
                )
            )

    # Compute final graph dimensions
    if result.nodes:
        min_x = min(n.x - n.width / 2 for n in result.nodes.values())
        max_x = max(n.x + n.width / 2 for n in result.nodes.values())
        min_y = min(n.y - n.height / 2 for n in result.nodes.values())
        max_y = max(n.y + n.height / 2 for n in result.nodes.values())

        # Shift everything so min is at 0
        for node in result.nodes.values():
            node.x -= min_x
            node.y -= min_y

        for edge in result.edges:
            edge.src_x -= min_x
            edge.src_y -= min_y
            edge.dst_x -= min_x
            edge.dst_y -= min_y

        result.width = max_x - min_x
        result.height = max_y - min_y

    return result
