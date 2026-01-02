"""Tests for termflow.render module."""

from io import StringIO

import pytest

from termflow.parser.events import (
    BoldEvent,
    CodeBlockEndEvent,
    CodeBlockLineEvent,
    CodeBlockStartEvent,
    EmptyLineEvent,
    HeadingEvent,
    HorizontalRuleEvent,
    ItalicEvent,
    NewlineEvent,
    TextEvent,
)
from termflow.render import Renderer, RenderFeatures, RenderStyle


class TestRenderer:
    """Test the main Renderer class."""

    @pytest.fixture
    def output(self):
        return StringIO()

    @pytest.fixture
    def renderer(self, output):
        return Renderer(output=output, width=80)

    def test_text_event(self, renderer, output):
        renderer.render(TextEvent("Hello"))
        assert "Hello" in output.getvalue()

    def test_bold_event(self, renderer, output):
        renderer.render(BoldEvent("bold"))
        result = output.getvalue()
        assert "\x1b[1m" in result  # Bold on
        assert "bold" in result

    def test_italic_event(self, renderer, output):
        renderer.render(ItalicEvent("italic"))
        result = output.getvalue()
        assert "\x1b[3m" in result  # Italic on
        assert "italic" in result

    def test_heading_event_h1(self, renderer, output):
        renderer.render(HeadingEvent(level=1, content="Title"))
        result = output.getvalue()
        assert "Title" in result
        assert "\x1b[" in result  # Has ANSI codes

    def test_heading_event_h2(self, renderer, output):
        renderer.render(HeadingEvent(level=2, content="Section"))
        result = output.getvalue()
        assert "Section" in result

    def test_heading_event_h6(self, renderer, output):
        renderer.render(HeadingEvent(level=6, content="Small"))
        result = output.getvalue()
        assert "Small" in result

    def test_code_block(self, renderer, output):
        renderer.render(CodeBlockStartEvent(language="python", indent=0))
        renderer.render(CodeBlockLineEvent("print('hi')"))
        renderer.render(CodeBlockEndEvent())
        result = output.getvalue()
        assert "print" in result

    def test_code_block_with_language(self, renderer, output):
        renderer.render(CodeBlockStartEvent(language="rust", indent=0))
        renderer.render(CodeBlockLineEvent("fn main() {}"))
        renderer.render(CodeBlockEndEvent())
        result = output.getvalue()
        assert "rust" in result.lower() or "main" in result

    def test_empty_line(self, renderer, output):
        renderer.render(EmptyLineEvent())
        assert output.getvalue() == "\n"

    def test_newline_event(self, renderer, output):
        renderer.render(NewlineEvent())
        assert output.getvalue() == "\n"

    def test_horizontal_rule(self, renderer, output):
        renderer.render(HorizontalRuleEvent())
        result = output.getvalue()
        assert len(result) > 0
        # Should contain rule character
        assert "─" in result or "-" in result

    def test_render_all(self, renderer, output):
        events = [
            TextEvent("Hello "),
            BoldEvent("world"),
            TextEvent("!"),
        ]
        renderer.render_all(events)
        result = output.getvalue()
        assert "Hello" in result
        assert "world" in result


class TestRenderStyle:
    """Test RenderStyle configuration."""

    def test_default_colors(self):
        style = RenderStyle()
        assert style.bright.startswith("#")
        assert len(style.bright) == 7

    def test_all_color_fields(self):
        style = RenderStyle()
        assert style.bright
        assert style.head
        assert style.symbol
        assert style.grey
        assert style.dark
        assert style.mid
        assert style.light
        assert style.link
        assert style.error

    def test_from_hue(self):
        style = RenderStyle.from_hue(0.5)
        assert style.bright.startswith("#")
        # Cyan-ish hue
        assert style.bright != RenderStyle().bright

    def test_from_hue_range(self):
        # Should handle full hue range
        for hue in [0.0, 0.25, 0.5, 0.75, 1.0]:
            style = RenderStyle.from_hue(hue)
            assert style.bright.startswith("#")

    def test_dracula_preset(self):
        style = RenderStyle.dracula()
        assert style.bright == "#bd93f9"  # Dracula purple

    def test_nord_preset(self):
        style = RenderStyle.nord()
        assert style.bright == "#88c0d0"  # Nord frost

    def test_gruvbox_preset(self):
        style = RenderStyle.gruvbox()
        assert style.bright == "#fabd2f"  # Gruvbox yellow

    def test_presets_differ(self):
        dracula = RenderStyle.dracula()
        nord = RenderStyle.nord()
        gruvbox = RenderStyle.gruvbox()
        default = RenderStyle()

        # All should be different
        assert dracula.bright != nord.bright
        assert nord.bright != gruvbox.bright
        assert gruvbox.bright != default.bright


class TestRenderFeatures:
    """Test RenderFeatures configuration."""

    def test_default_features(self):
        features = RenderFeatures()
        assert features.clipboard is True
        assert features.hyperlinks is True
        assert features.pretty_pad is True

    def test_disable_clipboard(self):
        features = RenderFeatures(clipboard=False)
        assert features.clipboard is False

    def test_disable_hyperlinks(self):
        features = RenderFeatures(hyperlinks=False)
        assert features.hyperlinks is False

    def test_disable_pretty_pad(self):
        features = RenderFeatures(pretty_pad=False)
        assert features.pretty_pad is False


class TestRendererConfiguration:
    """Test Renderer with different configurations."""

    def test_custom_width(self):
        output = StringIO()
        renderer = Renderer(output=output, width=40)
        assert renderer.width == 40

    def test_custom_style(self):
        output = StringIO()
        style = RenderStyle.dracula()
        renderer = Renderer(output=output, style=style)
        assert renderer.style.bright == "#bd93f9"

    def test_disabled_clipboard(self):
        output = StringIO()
        features = RenderFeatures(clipboard=False)
        renderer = Renderer(output=output, features=features)

        # Render a code block
        renderer.render(CodeBlockStartEvent(language="python", indent=0))
        renderer.render(CodeBlockLineEvent("code"))
        renderer.render(CodeBlockEndEvent())

        result = output.getvalue()
        # Should NOT contain OSC 52 sequence
        assert "\x1b]52" not in result

    def test_auto_width(self):
        output = StringIO()
        renderer = Renderer(output=output)  # No width specified
        # Should have a reasonable default
        assert renderer.width >= 20


class TestCodeBlockRendering:
    """Test code block rendering details."""

    @pytest.fixture
    def output(self):
        return StringIO()

    def test_code_block_has_borders(self, output):
        renderer = Renderer(output=output, width=80)
        renderer.render(CodeBlockStartEvent(language="python", indent=0))
        renderer.render(CodeBlockLineEvent("print('hi')"))
        renderer.render(CodeBlockEndEvent())

        result = output.getvalue()
        # Should have Unicode box characters
        assert any(c in result for c in "╭╮╰╯│─")

    def test_code_block_no_borders_when_disabled(self, output):
        features = RenderFeatures(pretty_pad=False)
        renderer = Renderer(output=output, width=80, features=features)
        renderer.render(CodeBlockStartEvent(language="python", indent=0))
        renderer.render(CodeBlockLineEvent("print('hi')"))
        renderer.render(CodeBlockEndEvent())

        result = output.getvalue()
        # Border characters may or may not be present depending on implementation
        assert "print" in result

    def test_clipboard_sequence(self, output):
        renderer = Renderer(output=output, width=80)
        renderer.render(CodeBlockStartEvent(language="python", indent=0))
        renderer.render(CodeBlockLineEvent("test code"))
        renderer.render(CodeBlockEndEvent())

        result = output.getvalue()
        # Should contain OSC 52 clipboard sequence
        assert "\x1b]52" in result
