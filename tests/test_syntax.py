"""Tests for termflow.syntax module."""

import pytest

from termflow.syntax import LANGUAGE_ALIASES, Highlighter, highlight_code


class TestHighlighter:
    """Test the Highlighter class."""

    @pytest.fixture
    def highlighter(self):
        return Highlighter()

    def test_highlight_python(self, highlighter):
        code = "def hello(): pass"
        result = highlighter.highlight_line(code, "python")
        assert "\x1b[" in result  # Has ANSI codes
        assert "def" in result

    def test_highlight_python_keywords(self, highlighter):
        code = "if True: return None"
        result = highlighter.highlight_line(code, "python")
        assert "\x1b[" in result

    def test_highlight_javascript(self, highlighter):
        code = "const x = () => 42;"
        result = highlighter.highlight_line(code, "javascript")
        assert "const" in result

    def test_highlight_rust(self, highlighter):
        code = 'fn main() { println!("Hi"); }'
        result = highlighter.highlight_line(code, "rust")
        assert "fn" in result

    def test_highlight_unknown_language(self, highlighter):
        code = "some text"
        result = highlighter.highlight_line(code, "unknown_lang_xyz_123")
        # Should still work (falls back to text)
        assert "some text" in result

    def test_language_aliases_python(self, highlighter):
        code = "print('hi')"
        result = highlighter.highlight_line(code, "py")
        assert "print" in result

    def test_language_aliases_javascript(self, highlighter):
        code = "console.log('hi')"
        result = highlighter.highlight_line(code, "js")
        assert "console" in result

    def test_language_aliases_typescript(self, highlighter):
        code = "const x: number = 42;"
        result = highlighter.highlight_line(code, "ts")
        assert "const" in result

    def test_get_lexer_valid(self, highlighter):
        lexer = highlighter.get_lexer("python")
        assert lexer is not None

    def test_get_lexer_alias(self, highlighter):
        lexer = highlighter.get_lexer("py")
        assert lexer is not None

    def test_get_lexer_invalid(self, highlighter):
        highlighter.get_lexer("nonexistent_language_xyz")
        # Should return None or text lexer
        # Implementation may vary

    def test_highlight_block(self, highlighter):
        code = 'fn main() {\n    println!("Hi");\n}'
        result = highlighter.highlight_block(code, "rust")
        assert "\x1b[" in result
        assert "main" in result

    def test_highlight_empty_string(self, highlighter):
        result = highlighter.highlight_line("", "python")
        # Should not crash
        assert isinstance(result, str)

    def test_highlight_multiline(self, highlighter):
        code = "def foo():\n    return 42"
        result = highlighter.highlight_block(code, "python")
        assert "def" in result
        assert "return" in result


class TestHighlighterStyles:
    """Test Highlighter with different Pygments styles."""

    def test_monokai_style(self):
        highlighter = Highlighter(style="monokai")
        result = highlighter.highlight_line("def x(): pass", "python")
        assert "\x1b[" in result

    def test_dracula_style(self):
        highlighter = Highlighter(style="dracula")
        result = highlighter.highlight_line("def x(): pass", "python")
        assert "\x1b[" in result

    def test_invalid_style_fallback(self):
        # Should fall back to default style
        highlighter = Highlighter(style="nonexistent_style_xyz")
        result = highlighter.highlight_line("def x(): pass", "python")
        # Should still work
        assert "def" in result

    def test_set_style(self):
        highlighter = Highlighter(style="monokai")
        highlighter.set_style("dracula")
        result = highlighter.highlight_line("def x(): pass", "python")
        assert "\x1b[" in result

    def test_available_styles(self):
        styles = Highlighter.available_styles()
        assert isinstance(styles, list)
        assert len(styles) > 0
        assert "monokai" in styles


class TestLanguageAliases:
    """Test language alias mapping."""

    def test_python_aliases(self):
        assert LANGUAGE_ALIASES.get("py") == "python"
        assert LANGUAGE_ALIASES.get("python3") == "python"

    def test_javascript_aliases(self):
        assert LANGUAGE_ALIASES.get("js") == "javascript"

    def test_typescript_aliases(self):
        assert LANGUAGE_ALIASES.get("ts") == "typescript"

    def test_rust_aliases(self):
        assert LANGUAGE_ALIASES.get("rs") == "rust"

    def test_shell_aliases(self):
        assert LANGUAGE_ALIASES.get("sh") == "bash"
        assert LANGUAGE_ALIASES.get("shell") == "bash"

    def test_cpp_aliases(self):
        assert LANGUAGE_ALIASES.get("c++") == "cpp"


class TestHighlightCode:
    """Test the convenience function."""

    def test_basic_usage(self):
        result = highlight_code("print('hi')", "python")
        assert "print" in result
        assert "\x1b[" in result

    def test_with_style(self):
        result = highlight_code("print('hi')", "python", style="monokai")
        assert "print" in result

    def test_unknown_language(self):
        result = highlight_code("some code", "unknown")
        assert "some code" in result

    def test_empty_code(self):
        result = highlight_code("", "python")
        assert isinstance(result, str)


class TestEdgeCases:
    """Test edge cases in syntax highlighting."""

    @pytest.fixture
    def highlighter(self):
        return Highlighter()

    def test_unicode_code(self, highlighter):
        code = 'print("你好世界")'
        result = highlighter.highlight_line(code, "python")
        assert "你好" in result

    def test_very_long_line(self, highlighter):
        code = "x = " + "1 + " * 1000 + "1"
        result = highlighter.highlight_line(code, "python")
        assert "x" in result

    def test_special_characters(self, highlighter):
        code = 'print("<>&\\"\'")'
        result = highlighter.highlight_line(code, "python")
        assert "print" in result

    def test_tabs_and_spaces(self, highlighter):
        code = "\tdef foo():\n        pass"
        result = highlighter.highlight_block(code, "python")
        assert "def" in result
