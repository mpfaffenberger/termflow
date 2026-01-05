"""Tests for the Mermaid layout module.

Comprehensive tests covering:
- Coordinate computation
- Direction transformations
- Graph dimensions
- Edge positioning
"""

import pytest

from termflow.render.mermaid import (
    Direction,
    Edge,
    EdgeStyle,
    GraphLayout,
    MermaidGraph,
    Node,
    NodeShape,
    layout_graph,
    parse_mermaid,
)


class TestMermaidLayoutBasic:
    """Basic layout tests."""

    def test_layout_empty_graph(self):
        """Layout empty graph returns empty layout."""
        graph = MermaidGraph()
        layout = layout_graph(graph)
        assert len(layout.nodes) == 0
        assert len(layout.edges) == 0
        assert layout.width == 0.0
        assert layout.height == 0.0

    def test_layout_single_node(self):
        """Layout single node graph."""
        graph = MermaidGraph(
            nodes={"A": Node("A", "Label")},
        )
        layout = layout_graph(graph)
        assert len(layout.nodes) == 1
        assert "A" in layout.nodes
        assert layout.nodes["A"].x >= 0
        assert layout.nodes["A"].y >= 0


class TestMermaidLayoutDirections:
    """Tests for direction-based layout."""

    def test_layout_linear_chain_lr(self):
        """Layout linear chain A → B → C in LR direction."""
        graph = MermaidGraph(
            direction=Direction.LR,
            nodes={
                "A": Node("A", "A"),
                "B": Node("B", "B"),
                "C": Node("C", "C"),
            },
            edges=[Edge("A", "B"), Edge("B", "C")],
        )
        layout = layout_graph(graph)

        # In LR direction, x should increase A < B < C
        assert layout.nodes["A"].x < layout.nodes["B"].x < layout.nodes["C"].x

    def test_layout_linear_chain_rl(self):
        """Layout linear chain in RL direction."""
        graph = MermaidGraph(
            direction=Direction.RL,
            nodes={
                "A": Node("A", "A"),
                "B": Node("B", "B"),
                "C": Node("C", "C"),
            },
            edges=[Edge("A", "B"), Edge("B", "C")],
        )
        layout = layout_graph(graph)

        # In RL direction, x should decrease A > B > C
        assert layout.nodes["A"].x > layout.nodes["B"].x > layout.nodes["C"].x

    def test_layout_linear_chain_tb(self):
        """Layout linear chain in TB direction."""
        graph = MermaidGraph(
            direction=Direction.TB,
            nodes={
                "A": Node("A", "A"),
                "B": Node("B", "B"),
                "C": Node("C", "C"),
            },
            edges=[Edge("A", "B"), Edge("B", "C")],
        )
        layout = layout_graph(graph)

        # In TB direction, y should increase A < B < C
        assert layout.nodes["A"].y < layout.nodes["B"].y < layout.nodes["C"].y

    def test_layout_linear_chain_bt(self):
        """Layout linear chain in BT direction."""
        graph = MermaidGraph(
            direction=Direction.BT,
            nodes={
                "A": Node("A", "A"),
                "B": Node("B", "B"),
                "C": Node("C", "C"),
            },
            edges=[Edge("A", "B"), Edge("B", "C")],
        )
        layout = layout_graph(graph)

        # In BT direction, y should decrease A > B > C
        assert layout.nodes["A"].y > layout.nodes["B"].y > layout.nodes["C"].y


class TestMermaidLayoutBranches:
    """Tests for graphs with branches."""

    def test_layout_with_branches(self):
        """Layout graph with branches."""
        code = """graph TD
            A --> B
            A --> C
            B --> D
            C --> D
        """
        graph = parse_mermaid(code)
        layout = layout_graph(graph)

        # All nodes should be positioned
        assert len(layout.nodes) == 4
        # D should be below B and C in TD direction
        assert layout.nodes["D"].y > layout.nodes["B"].y
        assert layout.nodes["D"].y > layout.nodes["C"].y


class TestMermaidLayoutProperties:
    """Tests for layout property preservation."""

    def test_layout_preserves_node_properties(self):
        """Layout preserves node label and shape."""
        graph = MermaidGraph(
            nodes={"A": Node("A", "My Label", NodeShape.DIAMOND)},
        )
        layout = layout_graph(graph)

        assert layout.nodes["A"].label == "My Label"
        assert layout.nodes["A"].shape == NodeShape.DIAMOND

    def test_layout_edges_have_endpoints(self):
        """Layout edges have source and destination coordinates."""
        graph = MermaidGraph(
            direction=Direction.LR,
            nodes={"A": Node("A", "A"), "B": Node("B", "B")},
            edges=[Edge("A", "B", label="Test", style=EdgeStyle.DOTTED)],
        )
        layout = layout_graph(graph)

        assert len(layout.edges) == 1
        edge = layout.edges[0]
        assert edge.src == "A"
        assert edge.dst == "B"
        assert edge.label == "Test"
        assert edge.style == EdgeStyle.DOTTED
        # Endpoints should be defined
        assert isinstance(edge.src_x, float)
        assert isinstance(edge.src_y, float)
        assert isinstance(edge.dst_x, float)
        assert isinstance(edge.dst_y, float)

    def test_layout_graph_dimensions(self):
        """Layout computes overall graph dimensions."""
        # Note: our parser requires one edge per line (chained edges not supported)
        graph = parse_mermaid("graph LR\n    A --> B\n    B --> C")
        layout = layout_graph(graph)

        assert layout.width > 0
        assert layout.height > 0
