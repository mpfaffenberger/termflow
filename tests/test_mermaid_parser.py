"""Tests for the Mermaid parser module.

Comprehensive tests covering:
- Syntax parsing
- Node shapes
- Edge styles
- Direction handling
- Error handling
"""

import pytest

from termflow.render.mermaid import (
    Direction,
    Edge,
    EdgeStyle,
    MermaidGraph,
    MermaidParseError,
    Node,
    NodeShape,
    is_mermaid_flowchart,
    parse_mermaid,
)


class TestMermaidParser:
    """Tests for Mermaid syntax parsing."""

    def test_parse_simple_flowchart(self):
        """Parse a basic flowchart with two nodes."""
        code = """graph LR
            A --> B
        """
        graph = parse_mermaid(code)
        assert graph.direction == Direction.LR
        assert "A" in graph.nodes
        assert "B" in graph.nodes
        assert len(graph.edges) == 1
        assert graph.edges[0].src == "A"
        assert graph.edges[0].dst == "B"

    def test_parse_direction_lr(self):
        """Parse left-to-right direction."""
        graph = parse_mermaid("graph LR\n    A --> B")
        assert graph.direction == Direction.LR

    def test_parse_direction_rl(self):
        """Parse right-to-left direction."""
        graph = parse_mermaid("graph RL\n    A --> B")
        assert graph.direction == Direction.RL

    def test_parse_direction_tb(self):
        """Parse top-to-bottom direction."""
        graph = parse_mermaid("graph TB\n    A --> B")
        assert graph.direction == Direction.TB

    def test_parse_direction_td(self):
        """Parse top-down direction (alias for TB)."""
        graph = parse_mermaid("graph TD\n    A --> B")
        assert graph.direction == Direction.TD

    def test_parse_direction_bt(self):
        """Parse bottom-to-top direction."""
        graph = parse_mermaid("graph BT\n    A --> B")
        assert graph.direction == Direction.BT

    def test_parse_flowchart_keyword(self):
        """Parse using 'flowchart' instead of 'graph'."""
        graph = parse_mermaid("flowchart LR\n    A --> B")
        assert graph.direction == Direction.LR
        assert len(graph.nodes) == 2


class TestMermaidParserNodeShapes:
    """Tests for node shape parsing."""

    def test_parse_node_rectangle(self):
        """Parse rectangle node shape [text]."""
        graph = parse_mermaid("graph TD\n    A[Rectangle Label]")
        assert graph.nodes["A"].shape == NodeShape.RECT
        assert graph.nodes["A"].label == "Rectangle Label"

    def test_parse_node_rounded(self):
        """Parse rounded node shape (text)."""
        graph = parse_mermaid("graph TD\n    A(Rounded Label)")
        assert graph.nodes["A"].shape == NodeShape.ROUNDED
        assert graph.nodes["A"].label == "Rounded Label"

    def test_parse_node_diamond(self):
        """Parse diamond node shape {text}."""
        graph = parse_mermaid("graph TD\n    A{Diamond Label}")
        assert graph.nodes["A"].shape == NodeShape.DIAMOND
        assert graph.nodes["A"].label == "Diamond Label"

    def test_parse_node_circle(self):
        """Parse circle node shape ((text))."""
        graph = parse_mermaid("graph TD\n    A((Circle Label))")
        assert graph.nodes["A"].shape == NodeShape.CIRCLE
        assert graph.nodes["A"].label == "Circle Label"

    def test_parse_node_flag(self):
        """Parse flag node shape >text]."""
        graph = parse_mermaid("graph TD\n    A>Flag Label]")
        assert graph.nodes["A"].shape == NodeShape.FLAG
        assert graph.nodes["A"].label == "Flag Label"

    def test_parse_all_node_shapes(self):
        """Parse all node shapes in one diagram."""
        code = """graph TD
            A[Rectangle]
            B(Rounded)
            C{Diamond}
            D((Circle))
            E>Flag]
        """
        graph = parse_mermaid(code)
        assert graph.nodes["A"].shape == NodeShape.RECT
        assert graph.nodes["B"].shape == NodeShape.ROUNDED
        assert graph.nodes["C"].shape == NodeShape.DIAMOND
        assert graph.nodes["D"].shape == NodeShape.CIRCLE
        assert graph.nodes["E"].shape == NodeShape.FLAG


class TestMermaidParserEdges:
    """Tests for edge parsing."""

    def test_parse_edge_solid_arrow(self):
        """Parse solid arrow edge -->."""
        graph = parse_mermaid("graph LR\n    A --> B")
        assert graph.edges[0].style == EdgeStyle.SOLID
        assert graph.edges[0].has_arrow is True

    def test_parse_edge_solid_line(self):
        """Parse solid line edge ---."""
        graph = parse_mermaid("graph LR\n    A --- B")
        assert graph.edges[0].style == EdgeStyle.SOLID
        assert graph.edges[0].has_arrow is False

    def test_parse_edge_dotted_arrow(self):
        """Parse dotted arrow edge -.->."""
        graph = parse_mermaid("graph LR\n    A -.-> B")
        assert graph.edges[0].style == EdgeStyle.DOTTED
        assert graph.edges[0].has_arrow is True

    def test_parse_edge_thick_arrow(self):
        """Parse thick arrow edge ==>."""
        graph = parse_mermaid("graph LR\n    A ==> B")
        assert graph.edges[0].style == EdgeStyle.THICK
        assert graph.edges[0].has_arrow is True

    def test_parse_edge_label(self):
        """Parse edge with label A -->|label| B."""
        graph = parse_mermaid("graph LR\n    A -->|Yes| B")
        assert graph.edges[0].label == "Yes"

    def test_parse_edge_label_with_spaces(self):
        """Parse edge label with spaces."""
        graph = parse_mermaid("graph LR\n    A -->|Some Label| B")
        assert graph.edges[0].label == "Some Label"


class TestMermaidParserNodes:
    """Tests for node handling."""

    def test_parse_implicit_nodes(self):
        """Parse nodes only defined in edges (implicit nodes)."""
        # Note: our parser handles one edge per line (chained edges not supported)
        graph = parse_mermaid("graph LR\n    A --> B\n    B --> C")
        assert "A" in graph.nodes
        assert "B" in graph.nodes
        assert "C" in graph.nodes
        # Implicit nodes should use ID as label
        assert graph.nodes["A"].label == "A"

    def test_parse_explicit_overrides_implicit(self):
        """Explicit node definition should override implicit."""
        code = """graph LR
            A --> B
            A[Explicit Label]
        """
        graph = parse_mermaid(code)
        assert graph.nodes["A"].label == "Explicit Label"


class TestMermaidParserMisc:
    """Tests for miscellaneous parsing features."""

    def test_parse_comments_ignored(self):
        """Lines starting with %% should be ignored."""
        code = """graph LR
            %% This is a comment
            A --> B
            %% Another comment
        """
        graph = parse_mermaid(code)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_parse_empty_lines_ignored(self):
        """Empty lines should be ignored."""
        code = """graph LR

            A --> B

            B --> C

        """
        graph = parse_mermaid(code)
        assert len(graph.edges) == 2

    def test_parse_case_insensitive_direction(self):
        """Direction should be case insensitive."""
        graph = parse_mermaid("graph lr\n    A --> B")
        assert graph.direction == Direction.LR


class TestMermaidParserErrors:
    """Tests for error handling."""

    def test_error_empty_content(self):
        """Empty content should raise error."""
        with pytest.raises(MermaidParseError):
            parse_mermaid("")

    def test_error_no_header(self):
        """Missing header should raise error."""
        with pytest.raises(MermaidParseError):
            parse_mermaid("A --> B")

    def test_error_invalid_node_id(self):
        """Invalid node ID should raise error."""
        with pytest.raises(MermaidParseError):
            parse_mermaid("graph LR\n    123 --> B")

    def test_error_whitespace_only(self):
        """Whitespace-only content should raise error."""
        with pytest.raises(MermaidParseError):
            parse_mermaid("   \n   \n   ")


class TestIsMermaidFlowchart:
    """Tests for is_mermaid_flowchart helper."""

    def test_is_mermaid_flowchart_positive(self):
        """is_mermaid_flowchart returns True for valid headers."""
        assert is_mermaid_flowchart("graph LR")
        assert is_mermaid_flowchart("flowchart TD")
        assert is_mermaid_flowchart("  graph TB  ")

    def test_is_mermaid_flowchart_negative(self):
        """is_mermaid_flowchart returns False for non-flowcharts."""
        assert not is_mermaid_flowchart("not mermaid")
        assert not is_mermaid_flowchart("sequenceDiagram")
        assert not is_mermaid_flowchart("")
