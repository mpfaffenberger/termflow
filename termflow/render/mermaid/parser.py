"""Mermaid flowchart syntax parser.

Parses Mermaid flowchart/graph syntax into structured data for rendering.

Supported features:
- Flowchart directions: LR, RL, TB, TD, BT
- Node shapes: rectangle [], rounded (), diamond {}, circle (()), flag >]
- Edge styles: solid -->, dotted -.->, thick ==>, line ---
- Edge labels: A -->|label| B
- Implicit nodes (defined only in edges)

Example:
    >>> parser = MermaidParser()
    >>> graph = parser.parse('''
    ...     graph LR
    ...         A[Start] --> B{Decision}
    ...         B -->|Yes| C[Process]
    ...         B -->|No| D((End))
    ... ''')
    >>> graph.direction
    Direction.LR
    >>> len(graph.nodes)
    4
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# =============================================================================
# Exceptions
# =============================================================================


class MermaidParseError(Exception):
    """Raised when Mermaid syntax cannot be parsed.

    The renderer can catch this to fall back to styled code display.
    """

    def __init__(self, message: str, line_number: int | None = None) -> None:
        self.line_number = line_number
        if line_number is not None:
            message = f"Line {line_number}: {message}"
        super().__init__(message)


# =============================================================================
# Enums
# =============================================================================


class Direction(Enum):
    """Flowchart direction."""

    LR = "LR"  # Left to Right
    RL = "RL"  # Right to Left
    TB = "TB"  # Top to Bottom
    TD = "TD"  # Top Down (same as TB)
    BT = "BT"  # Bottom to Top


class NodeShape(Enum):
    """Node shape type."""

    RECT = "rect"  # [text]
    ROUNDED = "rounded"  # (text)
    DIAMOND = "diamond"  # {text}
    CIRCLE = "circle"  # ((text))
    FLAG = "flag"  # >text]


class EdgeStyle(Enum):
    """Edge line style."""

    SOLID = "solid"  # --> or ---
    DOTTED = "dotted"  # -.->
    THICK = "thick"  # ==>


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Node:
    """A node in the flowchart.

    Attributes:
        id: Unique identifier for the node (used in edges)
        label: Display text for the node
        shape: Visual shape of the node
    """

    id: str
    label: str
    shape: NodeShape = NodeShape.RECT

    def __post_init__(self) -> None:
        """Validate node data."""
        if not self.id:
            raise ValueError("Node id cannot be empty")
        if not self.label:
            self.label = self.id  # Default label to id if empty


@dataclass
class Edge:
    """A connection between two nodes.

    Attributes:
        src: Source node ID
        dst: Destination node ID
        label: Optional text label on the edge
        style: Line style (solid, dotted, thick)
        has_arrow: Whether the edge has an arrowhead
    """

    src: str
    dst: str
    label: str | None = None
    style: EdgeStyle = EdgeStyle.SOLID
    has_arrow: bool = True

    def __post_init__(self) -> None:
        """Validate edge data."""
        if not self.src:
            raise ValueError("Edge source cannot be empty")
        if not self.dst:
            raise ValueError("Edge destination cannot be empty")


@dataclass
class MermaidGraph:
    """A parsed Mermaid flowchart/graph.

    Attributes:
        direction: Flow direction of the graph
        nodes: Dictionary of nodes by their ID
        edges: List of edges connecting nodes
    """

    direction: Direction = Direction.TB
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        If a node with the same ID exists, update it only if the new node
        has more information (e.g., explicit label vs implicit).
        """
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
        elif node.label != node.id or node.shape != NodeShape.RECT:
            # New node has explicit definition, prefer it
            self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def ensure_node(self, node_id: str) -> None:
        """Ensure a node exists, creating an implicit one if needed."""
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(id=node_id, label=node_id)


# =============================================================================
# Parser
# =============================================================================


class MermaidParser:
    """Parser for Mermaid flowchart syntax.

    Converts Mermaid text into structured MermaidGraph objects.

    Example:
        >>> parser = MermaidParser()
        >>> graph = parser.parse("graph LR\n    A --> B")
        >>> graph.direction
        Direction.LR
    """

    # Pattern for graph/flowchart header
    HEADER_PATTERN = re.compile(
        r"^\s*(?:graph|flowchart)\s+(LR|RL|TB|TD|BT)\s*$",
        re.IGNORECASE,
    )

    # Patterns for node shapes (order matters - check more specific first)
    # Circle: ((text))
    NODE_CIRCLE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(\((.+?)\)\)$")
    # Rounded: (text)
    NODE_ROUNDED = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(([^()]+)\)$")
    # Diamond: {text}
    NODE_DIAMOND = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\{(.+?)\}$")
    # Rectangle: [text]
    NODE_RECT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[([^\]]+)\]$")
    # Flag/Asymmetric: >text]
    NODE_FLAG = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)>([^\]]+)\]$")
    # Plain node ID (no shape)
    NODE_PLAIN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)$")

    # Edge patterns (order matters - check longer patterns first)
    EDGE_PATTERNS: list[tuple[re.Pattern[str], EdgeStyle, bool]] = [
        # Dotted arrow: -.->
        (re.compile(r"^-\.->$"), EdgeStyle.DOTTED, True),
        # Dotted line: -.-
        (re.compile(r"^-\.-$"), EdgeStyle.DOTTED, False),
        # Thick arrow: ==>
        (re.compile(r"^==>$"), EdgeStyle.THICK, True),
        # Thick line: ===
        (re.compile(r"^===$"), EdgeStyle.THICK, False),
        # Solid arrow: -->
        (re.compile(r"^-->$"), EdgeStyle.SOLID, True),
        # Solid line: ---
        (re.compile(r"^---$"), EdgeStyle.SOLID, False),
    ]

    # Pattern to extract edge with optional label
    # Matches: NODE_SPEC EDGE_OP |label| NODE_SPEC
    # or:      NODE_SPEC EDGE_OP NODE_SPEC
    EDGE_LINE_PATTERN = re.compile(
        r"^\s*"
        r"(.+?)"  # Source node (non-greedy)
        r"\s+"
        r"(-->|---|-\.->|==>|===|-\.-)"  # Edge operator
        r"(?:\|([^|]*)\|)?"  # Optional label in |pipes|
        r"\s+"
        r"(.+?)"  # Destination node (non-greedy)
        r"\s*$"
    )

    def __init__(self) -> None:
        """Initialize the parser."""
        pass

    def parse(self, content: str) -> MermaidGraph:
        """Parse Mermaid flowchart content into a MermaidGraph.

        Args:
            content: Mermaid flowchart text

        Returns:
            Parsed MermaidGraph structure

        Raises:
            MermaidParseError: If the content cannot be parsed
        """
        lines = content.strip().splitlines()
        if not lines:
            raise MermaidParseError("Empty mermaid content")

        graph = MermaidGraph()
        header_found = False

        for line_num, line in enumerate(lines, start=1):
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue

            # Try to parse header
            if not header_found:
                direction = self._parse_header(stripped)
                if direction is not None:
                    graph.direction = direction
                    header_found = True
                    continue
                # First non-empty line should be header
                raise MermaidParseError(
                    f"Expected 'graph' or 'flowchart' header, got: {stripped!r}",
                    line_number=line_num,
                )

            # Try to parse node definition or edge
            try:
                self._parse_line(stripped, graph, line_num)
            except MermaidParseError:
                raise
            except Exception as e:
                raise MermaidParseError(str(e), line_number=line_num) from e

        if not header_found:
            raise MermaidParseError("No flowchart header found")

        return graph

    def _parse_header(self, line: str) -> Direction | None:
        """Parse a flowchart header line.

        Args:
            line: Line to parse

        Returns:
            Direction if header is found, None otherwise
        """
        match = self.HEADER_PATTERN.match(line)
        if match:
            direction_str = match.group(1).upper()
            return Direction(direction_str)
        return None

    def _parse_line(self, line: str, graph: MermaidGraph, line_num: int) -> None:
        """Parse a single line of flowchart content.

        Args:
            line: Line to parse (already stripped)
            graph: Graph to add parsed elements to
            line_num: Line number for error reporting
        """
        # Try to parse as edge (most common)
        edge_match = self.EDGE_LINE_PATTERN.match(line)
        if edge_match:
            src_spec = edge_match.group(1).strip()
            edge_op = edge_match.group(2)
            edge_label = edge_match.group(3)  # May be None
            dst_spec = edge_match.group(4).strip()

            # Parse source node
            src_node = self._parse_node_spec(src_spec)
            if src_node is None:
                raise MermaidParseError(
                    f"Invalid source node: {src_spec!r}",
                    line_number=line_num,
                )
            graph.add_node(src_node)

            # Parse destination node
            dst_node = self._parse_node_spec(dst_spec)
            if dst_node is None:
                raise MermaidParseError(
                    f"Invalid destination node: {dst_spec!r}",
                    line_number=line_num,
                )
            graph.add_node(dst_node)

            # Parse edge style
            style, has_arrow = self._parse_edge_operator(edge_op)

            # Create and add edge
            edge = Edge(
                src=src_node.id,
                dst=dst_node.id,
                label=edge_label.strip() if edge_label else None,
                style=style,
                has_arrow=has_arrow,
            )
            graph.add_edge(edge)
            return

        # Try to parse as standalone node definition
        node = self._parse_node_spec(line)
        if node is not None:
            graph.add_node(node)
            return

        # If we get here, we couldn't parse the line
        raise MermaidParseError(
            f"Could not parse line: {line!r}",
            line_number=line_num,
        )

    def _parse_node_spec(self, spec: str) -> Node | None:
        """Parse a node specification into a Node object.

        Args:
            spec: Node specification (e.g., "A", "A[Label]", "B{Decision}")

        Returns:
            Node if parsing succeeded, None otherwise
        """
        spec = spec.strip()
        if not spec:
            return None

        # Try each shape pattern in order of specificity
        # Circle: ((text))
        match = self.NODE_CIRCLE.match(spec)
        if match:
            return Node(
                id=match.group(1),
                label=match.group(2).strip(),
                shape=NodeShape.CIRCLE,
            )

        # Rounded: (text)
        match = self.NODE_ROUNDED.match(spec)
        if match:
            return Node(
                id=match.group(1),
                label=match.group(2).strip(),
                shape=NodeShape.ROUNDED,
            )

        # Diamond: {text}
        match = self.NODE_DIAMOND.match(spec)
        if match:
            return Node(
                id=match.group(1),
                label=match.group(2).strip(),
                shape=NodeShape.DIAMOND,
            )

        # Flag: >text]
        match = self.NODE_FLAG.match(spec)
        if match:
            return Node(
                id=match.group(1),
                label=match.group(2).strip(),
                shape=NodeShape.FLAG,
            )

        # Rectangle: [text]
        match = self.NODE_RECT.match(spec)
        if match:
            return Node(
                id=match.group(1),
                label=match.group(2).strip(),
                shape=NodeShape.RECT,
            )

        # Plain node ID
        match = self.NODE_PLAIN.match(spec)
        if match:
            node_id = match.group(1)
            return Node(id=node_id, label=node_id)

        return None

    def _parse_edge_operator(self, operator: str) -> tuple[EdgeStyle, bool]:
        """Parse an edge operator into style and arrow info.

        Args:
            operator: Edge operator string (e.g., "-->", "---", "-.->", "==>")

        Returns:
            Tuple of (EdgeStyle, has_arrow)
        """
        for pattern, style, has_arrow in self.EDGE_PATTERNS:
            if pattern.match(operator):
                return style, has_arrow

        # Default to solid arrow if unknown
        return EdgeStyle.SOLID, True


# =============================================================================
# Convenience Functions
# =============================================================================


def parse_mermaid(content: str) -> MermaidGraph:
    """Parse Mermaid flowchart content.

    Convenience function that creates a parser and parses content.

    Args:
        content: Mermaid flowchart text

    Returns:
        Parsed MermaidGraph

    Raises:
        MermaidParseError: If parsing fails
    """
    parser = MermaidParser()
    return parser.parse(content)


def is_mermaid_flowchart(content: str) -> bool:
    """Check if content looks like a Mermaid flowchart.

    Quick check without full parsing - useful for detection.

    Args:
        content: Content to check

    Returns:
        True if content starts with flowchart/graph header
    """
    first_line = content.strip().split("\n", 1)[0].strip().lower()
    return first_line.startswith(("graph ", "flowchart "))
