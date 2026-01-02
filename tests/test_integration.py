"""Integration tests for termflow."""

from io import StringIO

import pytest

from termflow import Config, Parser, Renderer, RenderStyle, render_markdown


class TestFullPipeline:
    """Test the complete parsing and rendering pipeline."""

    def test_simple_document(self):
        markdown = """# Hello World

This is a **bold** and *italic* test.

```python
print("Hello!")
```

- Item 1
- Item 2

> A quote
"""
        output = StringIO()
        parser = Parser()
        renderer = Renderer(output=output, width=80)

        events = parser.parse_document(markdown)
        renderer.render_all(events)

        result = output.getvalue()
        assert "Hello World" in result
        assert "print" in result
        assert "quote" in result

    def test_code_block_highlighting(self):
        markdown = """```python
def greet(name):
    return f"Hello, {name}!"
```"""
        output = StringIO()
        parser = Parser()
        renderer = Renderer(output=output, width=80)

        events = parser.parse_document(markdown)
        renderer.render_all(events)

        result = output.getvalue()
        assert "def" in result
        assert "greet" in result
        # Should have ANSI coloring
        assert "\x1b[" in result

    def test_table_rendering(self):
        markdown = """| Name | Age |
|------|-----|
| Alice | 30 |
| Bob | 25 |
"""
        output = StringIO()
        parser = Parser()
        renderer = Renderer(output=output, width=80)

        events = parser.parse_document(markdown)
        renderer.render_all(events)

        result = output.getvalue()
        assert "Name" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_nested_lists(self):
        markdown = """- Parent 1
  - Child 1a
  - Child 1b
- Parent 2
  - Child 2a
"""
        output = StringIO()
        parser = Parser()
        renderer = Renderer(output=output, width=80)

        events = parser.parse_document(markdown)
        renderer.render_all(events)

        result = output.getvalue()
        assert "Parent" in result
        assert "Child" in result

    def test_think_blocks(self):
        markdown = """<think>
Let me reason about this...
Step 1: Analyze the problem
Step 2: Formulate solution
</think>

Here is my answer.
"""
        output = StringIO()
        parser = Parser()
        renderer = Renderer(output=output, width=80)

        events = parser.parse_document(markdown)
        renderer.render_all(events)

        result = output.getvalue()
        assert "reason" in result
        assert "answer" in result


class TestRenderMarkdown:
    """Test the render_markdown convenience function."""

    def test_basic_usage(self):
        output = StringIO()
        render_markdown("# Test", output=output)
        assert "Test" in output.getvalue()

    def test_with_width(self):
        output = StringIO()
        render_markdown("# Test", output=output, width=40)
        result = output.getvalue()
        assert "Test" in result

    def test_with_style(self):
        output = StringIO()
        style = RenderStyle.dracula()
        render_markdown("# Test", output=output, style=style)
        result = output.getvalue()
        assert "Test" in result
        # Should have Dracula colors
        assert "\x1b[" in result

    def test_complex_document(self):
        markdown = """# Title

Paragraph with **bold** and `code`.

## Section

1. First
2. Second
3. Third

---

The end.
"""
        output = StringIO()
        render_markdown(markdown, output=output, width=80)
        result = output.getvalue()

        assert "Title" in result
        assert "Section" in result
        assert "First" in result


class TestStreaming:
    """Test streaming (line-by-line) parsing and rendering."""

    def test_line_by_line(self):
        lines = [
            "# Title",
            "",
            "Paragraph text.",
            "",
            "```",
            "code",
            "```",
        ]

        output = StringIO()
        parser = Parser()
        renderer = Renderer(output=output, width=80)

        for line in lines:
            events = parser.parse_line(line)
            renderer.render_all(events)

        renderer.render_all(parser.finalize())

        result = output.getvalue()
        assert "Title" in result
        assert "code" in result

    def test_streaming_code_block(self):
        lines = [
            "```python",
            "def foo():",
            "    return 42",
            "```",
        ]

        output = StringIO()
        parser = Parser()
        renderer = Renderer(output=output, width=80)

        for line in lines:
            events = parser.parse_line(line)
            renderer.render_all(events)

        renderer.render_all(parser.finalize())

        result = output.getvalue()
        assert "def" in result
        assert "42" in result

    def test_streaming_with_interruption(self):
        """Test that parser state is maintained across lines."""
        parser = Parser()
        renderer_output = StringIO()
        renderer = Renderer(output=renderer_output, width=80)

        # Start code block
        events = parser.parse_line("```python")
        renderer.render_all(events)

        # Add some lines
        events = parser.parse_line("x = 1")
        renderer.render_all(events)
        events = parser.parse_line("y = 2")
        renderer.render_all(events)

        # End code block
        events = parser.parse_line("```")
        renderer.render_all(events)

        result = renderer_output.getvalue()
        # Code lines are syntax highlighted, check for key parts
        from termflow.ansi import visible

        visible_result = visible(result)
        assert "x" in visible_result or "1" in visible_result
        assert "y" in visible_result or "2" in visible_result


class TestConfig:
    """Test configuration integration."""

    def test_default_config(self):
        config = Config()
        assert config.max_width == 120
        assert config.syntax_style == "monokai"

    def test_config_style(self):
        config = Config()
        assert config.style is not None
        assert config.style.bright.startswith("#")

    def test_config_features(self):
        config = Config()
        assert config.features is not None
        assert config.features.clipboard is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_document(self):
        output = StringIO()
        render_markdown("", output=output)
        # Should not crash
        assert isinstance(output.getvalue(), str)

    def test_only_whitespace(self):
        output = StringIO()
        render_markdown("   \n\n   ", output=output)
        # Should not crash
        assert isinstance(output.getvalue(), str)

    def test_very_long_line(self):
        long_line = "word " * 1000
        output = StringIO()
        render_markdown(long_line, output=output, width=80)
        # Should handle gracefully
        assert len(output.getvalue()) > 0

    def test_unicode_content(self):
        markdown = """# 你好世界

こんにちは!

- 한국어
- Русский
"""
        output = StringIO()
        render_markdown(markdown, output=output, width=80)
        result = output.getvalue()
        assert "你好" in result
        assert "こんにちは" in result

    def test_special_characters(self):
        markdown = """Special chars: <>&"'

Code: `<html>&amp;</html>`
"""
        output = StringIO()
        render_markdown(markdown, output=output)
        result = output.getvalue()
        assert "<" in result or "&lt;" in result

    def test_deeply_nested_content(self):
        markdown = """>>> Deep quote
>>>> Even deeper
>>>>> Deepest
"""
        output = StringIO()
        parser = Parser()
        renderer = Renderer(output=output, width=80)

        events = parser.parse_document(markdown)
        renderer.render_all(events)
        # Should handle gracefully
        assert len(output.getvalue()) > 0


class TestStylePresets:
    """Test rendering with different style presets."""

    @pytest.mark.parametrize(
        "style_factory",
        [
            RenderStyle,
            RenderStyle.dracula,
            RenderStyle.nord,
            RenderStyle.gruvbox,
        ],
    )
    def test_style_rendering(self, style_factory):
        markdown = """# Heading

**Bold** and *italic*.

```python
print("hello")
```
"""
        output = StringIO()
        style = style_factory()
        render_markdown(markdown, output=output, style=style)

        result = output.getvalue()
        assert "Heading" in result
        assert "Bold" in result
        assert "print" in result
