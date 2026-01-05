"""Integration tests for the Mermaid rendering system.

Comprehensive tests covering:
- Full pipeline from code to terminal output
- Fallback behavior
- Feature flags
- Renderer integration
"""

import pytest

from termflow.render.mermaid import (
    GraphicsProtocol,
    MermaidParseError,
    parse_mermaid,
    render_mermaid_to_terminal,
)


class TestMermaidPipeline:
    """Tests for full pipeline integration."""

    def test_full_pipeline_simple(self):
        """Full pipeline from code to terminal output."""
        code = """graph LR
            A[Start] --> B[End]
        """
        output = render_mermaid_to_terminal(code, width=80)

        assert isinstance(output, str)
        assert len(output) > 0

    def test_full_pipeline_complex(self):
        """Full pipeline with complex diagram."""
        code = """graph TD
            A[Start] --> B{Decision}
            B -->|Yes| C[Process]
            B -->|No| D[Skip]
            C --> E((End))
            D --> E
        """
        output = render_mermaid_to_terminal(code, width=80)

        assert isinstance(output, str)
        assert len(output) > 0

    def test_full_pipeline_with_protocol(self):
        """Full pipeline with explicit protocol."""
        code = "graph LR\n    A --> B"

        block_output = render_mermaid_to_terminal(
            code, width=60, protocol=GraphicsProtocol.BLOCK
        )
        assert isinstance(block_output, str)
        # Block output should contain block characters or ANSI codes
        assert "\x1b[" in block_output or "█" in block_output


class TestMermaidErrorHandling:
    """Tests for error handling."""

    def test_invalid_syntax_raises_error(self):
        """Invalid mermaid syntax raises MermaidParseError."""
        with pytest.raises(MermaidParseError):
            parse_mermaid("this is not valid mermaid")

    def test_invalid_syntax_in_pipeline(self):
        """Invalid syntax in render_mermaid_to_terminal raises error."""
        with pytest.raises(MermaidParseError):
            render_mermaid_to_terminal("invalid syntax")

    def test_parse_mermaid_error_has_line_number(self):
        """MermaidParseError includes line number when available."""
        try:
            parse_mermaid("graph LR\n    123invalid --> B")
            assert False, "Should have raised MermaidParseError"
        except MermaidParseError as e:
            assert e.line_number is not None
            assert "Line" in str(e)


class TestMermaidRendererIntegration:
    """Tests for integration with main termflow renderer."""

    def test_renderer_handles_mermaid_block(self):
        """Renderer processes mermaid code blocks."""
        from io import StringIO

        from termflow.parser import Parser
        from termflow.render import Renderer

        markdown = """```mermaid
graph LR
    A --> B
```"""
        output = StringIO()
        parser = Parser()
        renderer = Renderer(output=output, width=60)

        events = parser.parse_document(markdown)
        renderer.render_all(events)

        result = output.getvalue()
        assert len(result) > 0

    def test_renderer_fallback_on_invalid_mermaid(self):
        """Renderer falls back to code block on invalid mermaid."""
        from io import StringIO

        from termflow.parser import Parser
        from termflow.render import Renderer

        markdown = """```mermaid
this is not valid mermaid
```"""
        output = StringIO()
        parser = Parser()
        renderer = Renderer(output=output, width=60)

        events = parser.parse_document(markdown)
        renderer.render_all(events)

        result = output.getvalue()
        # Should contain the fallback with mermaid label
        assert len(result) > 0
        # Should contain the original code (fallback mode)
        assert "not valid" in result or "mermaid" in result.lower()

    def test_renderer_respects_mermaid_graphics_flag(self):
        """Renderer respects mermaid_graphics=False."""
        from io import StringIO

        from termflow.parser import Parser
        from termflow.render import Renderer, RenderFeatures

        markdown = """```mermaid
graph LR
    A --> B
```"""
        output = StringIO()
        features = RenderFeatures(mermaid_graphics=False)
        parser = Parser()
        renderer = Renderer(output=output, width=60, features=features)

        events = parser.parse_document(markdown)
        renderer.render_all(events)

        result = output.getvalue()
        # Should contain the mermaid code as text (not rendered as graphics)
        assert "graph LR" in result or "mermaid" in result.lower()

    def test_renderer_reset_clears_mermaid_state(self):
        """Renderer.reset() clears mermaid state."""
        from termflow.render import Renderer

        renderer = Renderer(width=60)
        renderer._mermaid_mode = True
        renderer._mermaid_buffer = ["some", "content"]

        renderer.reset()

        assert renderer._mermaid_mode is False
        assert renderer._mermaid_buffer == []
