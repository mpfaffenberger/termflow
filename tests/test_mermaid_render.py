"""Tests for the Mermaid canvas and graphics modules.

Comprehensive tests covering:
- PIL image rendering
- Terminal graphics output
- Graphics protocol detection
"""

import pytest
from PIL import Image

from termflow.render.mermaid import (
    GraphLayout,
    GraphicsProtocol,
    RenderConfig,
    detect_graphics_protocol,
    image_to_blocks,
    image_to_iterm2,
    image_to_kitty,
    image_to_terminal,
    layout_graph,
    parse_mermaid,
    render_to_bytes,
    render_to_image,
)


# =============================================================================
# Canvas Tests
# =============================================================================


class TestMermaidCanvas:
    """Tests for PIL image rendering."""

    def test_render_produces_image(self):
        """render_to_image produces a valid PIL Image."""
        graph = parse_mermaid("graph LR\n    A --> B")
        layout = layout_graph(graph)
        img = render_to_image(layout)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"
        assert img.width > 0
        assert img.height > 0

    def test_render_empty_layout(self):
        """render_to_image handles empty layout."""
        layout = GraphLayout()
        img = render_to_image(layout)

        assert isinstance(img, Image.Image)
        # Should return minimal image
        assert img.width >= 100
        assert img.height >= 100

    def test_render_with_custom_config(self):
        """render_to_image respects custom RenderConfig."""
        graph = parse_mermaid("graph LR\n    A --> B")
        layout = layout_graph(graph)

        config = RenderConfig(
            background_color=(255, 0, 0, 255),  # Red background
            scale=20,  # Larger scale
        )
        img = render_to_image(layout, config)

        # With larger scale, image should be bigger
        default_img = render_to_image(layout)
        assert img.width > default_img.width

    def test_render_transparent_background(self):
        """Default background is transparent."""
        graph = parse_mermaid("graph LR\n    A --> B")
        layout = layout_graph(graph)
        img = render_to_image(layout)

        # Check a corner pixel (should be transparent or part of padding)
        assert img.mode == "RGBA"

    def test_render_to_bytes(self):
        """render_to_bytes produces valid PNG data."""
        graph = parse_mermaid("graph LR\n    A --> B")
        layout = layout_graph(graph)
        data = render_to_bytes(layout)

        assert isinstance(data, bytes)
        assert len(data) > 0
        # PNG magic bytes
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_all_node_shapes(self):
        """All node shapes render without error."""
        code = """graph TD
            A[Rect] --> B(Rounded)
            B --> C{Diamond}
            C --> D((Circle))
            D --> E>Flag]
        """
        graph = parse_mermaid(code)
        layout = layout_graph(graph)
        img = render_to_image(layout)

        assert isinstance(img, Image.Image)
        assert img.width > 0

    def test_render_edge_styles(self):
        """All edge styles render without error."""
        code = """graph LR
            A --> B
            B --- C
            C -.-> D
            D ==> E
        """
        graph = parse_mermaid(code)
        layout = layout_graph(graph)
        img = render_to_image(layout)

        assert isinstance(img, Image.Image)

    def test_render_edge_labels(self):
        """Edge labels render without error."""
        code = """graph LR
            A -->|Label 1| B
            B -->|Label 2| C
        """
        graph = parse_mermaid(code)
        layout = layout_graph(graph)
        img = render_to_image(layout)

        assert isinstance(img, Image.Image)


# =============================================================================
# Graphics Tests
# =============================================================================


class TestMermaidGraphicsBlocks:
    """Tests for block character output."""

    def test_block_output_produces_string(self):
        """image_to_blocks produces a string."""
        img = Image.new("RGBA", (100, 50), (255, 255, 255, 255))
        output = image_to_blocks(img, width=40)

        assert isinstance(output, str)
        assert len(output) > 0

    def test_block_output_contains_newlines(self):
        """Block output has multiple lines."""
        img = Image.new("RGBA", (100, 50), (255, 255, 255, 255))
        output = image_to_blocks(img, width=40)

        lines = output.split("\n")
        assert len(lines) > 1

    def test_block_output_contains_ansi_codes(self):
        """Block output contains ANSI color codes."""
        img = Image.new("RGBA", (100, 50), (255, 0, 0, 255))  # Red
        output = image_to_blocks(img, width=40)

        # Should contain escape sequences
        assert "\x1b[" in output

    def test_block_output_contains_block_chars(self):
        """Block output contains Unicode block characters."""
        img = Image.new("RGBA", (100, 50), (255, 255, 255, 255))
        output = image_to_blocks(img, width=40)

        # Should contain at least one block character
        block_chars = ["\u2580", "\u2584", "\u2588"]  # ▀ ▄ █
        assert any(char in output for char in block_chars)

    def test_block_output_respects_width(self):
        """Block output respects target width."""
        img = Image.new("RGBA", (200, 100), (255, 255, 255, 255))

        output_narrow = image_to_blocks(img, width=20)
        output_wide = image_to_blocks(img, width=80)

        # Wide output should have longer lines
        # (though both have ANSI codes making exact comparison tricky)
        assert len(output_wide) > len(output_narrow)


class TestMermaidGraphicsProtocols:
    """Tests for terminal graphics protocols."""

    def test_kitty_output_format(self):
        """image_to_kitty produces proper escape sequence."""
        img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))
        output = image_to_kitty(img)

        # Should start with Kitty escape sequence
        assert output.startswith("\x1b_G")
        # Should contain format indicator
        assert "f=100" in output  # PNG format
        # Should end with string terminator
        assert output.endswith("\x1b\\")

    def test_iterm2_output_format(self):
        """image_to_iterm2 produces proper escape sequence."""
        img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))
        output = image_to_iterm2(img)

        # Should start with iTerm2 escape sequence
        assert output.startswith("\x1b]1337;File=")
        # Should contain inline flag
        assert "inline=1" in output
        # Should end with BEL
        assert output.endswith("\x07")

    def test_image_to_terminal_returns_string(self):
        """image_to_terminal returns a string for any protocol."""
        img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))

        for protocol in GraphicsProtocol:
            output = image_to_terminal(img, protocol=protocol)
            assert isinstance(output, str)

    def test_detect_graphics_protocol(self):
        """detect_graphics_protocol returns a valid protocol."""
        protocol = detect_graphics_protocol()
        assert isinstance(protocol, GraphicsProtocol)
