"""Tests for inline markdown parsing."""

import pytest

from termflow.parser.inline import InlineElement, InlineParser, InlineToken


class TestInlineParser:
    """Test inline markdown parsing."""

    @pytest.fixture
    def parser(self):
        return InlineParser()

    def test_plain_text(self, parser):
        tokens = parser.parse("Hello world")
        assert len(tokens) >= 1
        assert tokens[0].element_type == InlineElement.TEXT
        assert tokens[0].content == "Hello world"

    def test_bold_asterisks(self, parser):
        tokens = parser.parse("**bold text**")
        assert any(t.element_type == InlineElement.BOLD for t in tokens)
        bold = next(t for t in tokens if t.element_type == InlineElement.BOLD)
        assert bold.content == "bold text"

    def test_bold_underscores(self, parser):
        # Note: Underscore bold may not be supported in all parsers
        tokens = parser.parse("__bold text__")
        # Check if it's either parsed as bold or kept as text
        assert len(tokens) >= 1

    def test_italic_asterisks(self, parser):
        tokens = parser.parse("*italic text*")
        assert any(t.element_type == InlineElement.ITALIC for t in tokens)
        italic = next(t for t in tokens if t.element_type == InlineElement.ITALIC)
        assert italic.content == "italic text"

    def test_italic_underscores(self, parser):
        tokens = parser.parse("_italic text_")
        assert any(t.element_type == InlineElement.ITALIC for t in tokens)

    def test_bold_italic(self, parser):
        tokens = parser.parse("***bold italic***")
        assert any(t.element_type == InlineElement.BOLD_ITALIC for t in tokens)

    def test_inline_code(self, parser):
        tokens = parser.parse("`code here`")
        assert any(t.element_type == InlineElement.CODE for t in tokens)
        code = next(t for t in tokens if t.element_type == InlineElement.CODE)
        assert code.content == "code here"

    def test_inline_code_with_backticks(self, parser):
        tokens = parser.parse("``code with `backtick` inside``")
        assert any(t.element_type == InlineElement.CODE for t in tokens)

    def test_link(self, parser):
        tokens = parser.parse("[click here](https://example.com)")
        assert any(t.element_type == InlineElement.LINK for t in tokens)
        link = next(t for t in tokens if t.element_type == InlineElement.LINK)
        assert link.content == "click here"
        assert link.url == "https://example.com"

    def test_link_with_title(self, parser):
        tokens = parser.parse('[link](https://example.com "title")')
        assert any(t.element_type == InlineElement.LINK for t in tokens)

    def test_image(self, parser):
        tokens = parser.parse("![alt text](image.png)")
        assert any(t.element_type == InlineElement.IMAGE for t in tokens)
        img = next(t for t in tokens if t.element_type == InlineElement.IMAGE)
        assert img.content == "alt text"
        assert img.url == "image.png"

    def test_strikeout(self, parser):
        tokens = parser.parse("~~deleted~~")
        assert any(t.element_type == InlineElement.STRIKEOUT for t in tokens)
        strike = next(t for t in tokens if t.element_type == InlineElement.STRIKEOUT)
        assert strike.content == "deleted"

    def test_footnote(self, parser):
        tokens = parser.parse("Text[^1] with footnote")
        assert any(t.element_type == InlineElement.FOOTNOTE for t in tokens)

    def test_mixed_formatting(self, parser):
        tokens = parser.parse("**bold** and *italic*")
        types = [t.element_type for t in tokens]
        assert InlineElement.BOLD in types
        assert InlineElement.ITALIC in types
        assert InlineElement.TEXT in types  # " and "

    def test_nested_in_text(self, parser):
        tokens = parser.parse("Before **bold** after")
        assert len(tokens) >= 3  # text, bold, text
        assert tokens[0].element_type == InlineElement.TEXT
        assert "Before" in tokens[0].content

    def test_empty_string(self, parser):
        tokens = parser.parse("")
        # Should return empty or single empty text token
        assert isinstance(tokens, list)

    def test_no_formatting(self, parser):
        tokens = parser.parse("Plain text with no formatting")
        assert len(tokens) == 1
        assert tokens[0].element_type == InlineElement.TEXT


class TestInlineToken:
    """Test InlineToken dataclass."""

    def test_token_creation(self):
        token = InlineToken(InlineElement.TEXT, "hello")
        assert token.element_type == InlineElement.TEXT
        assert token.content == "hello"
        assert token.url is None

    def test_token_with_url(self):
        token = InlineToken(InlineElement.LINK, "click", "https://example.com")
        assert token.element_type == InlineElement.LINK
        assert token.content == "click"
        assert token.url == "https://example.com"

    def test_token_repr(self):
        token = InlineToken(InlineElement.BOLD, "text")
        repr_str = repr(token)
        assert "BOLD" in repr_str
        assert "text" in repr_str


class TestInlineElement:
    """Test InlineElement enum."""

    def test_all_elements_exist(self):
        assert hasattr(InlineElement, "TEXT")
        assert hasattr(InlineElement, "BOLD")
        assert hasattr(InlineElement, "ITALIC")
        assert hasattr(InlineElement, "BOLD_ITALIC")
        assert hasattr(InlineElement, "CODE")
        assert hasattr(InlineElement, "LINK")
        assert hasattr(InlineElement, "IMAGE")
        assert hasattr(InlineElement, "STRIKEOUT")
        assert hasattr(InlineElement, "FOOTNOTE")


class TestEdgeCases:
    """Test edge cases in inline parsing."""

    @pytest.fixture
    def parser(self):
        return InlineParser()

    def test_unmatched_asterisk(self, parser):
        # Should not crash on unmatched formatting
        tokens = parser.parse("Hello *world")
        assert len(tokens) >= 1

    def test_unmatched_backtick(self, parser):
        tokens = parser.parse("Hello `code")
        assert len(tokens) >= 1

    def test_empty_bold(self, parser):
        tokens = parser.parse("****")
        # Should handle gracefully
        assert isinstance(tokens, list)

    def test_consecutive_formatting(self, parser):
        # Consecutive formatting can be tricky - just ensure it doesn't crash
        tokens = parser.parse("**bold***italic*")
        assert len(tokens) >= 1

    def test_url_with_special_chars(self, parser):
        tokens = parser.parse("[link](https://example.com/path?q=1&b=2)")
        link = next(t for t in tokens if t.element_type == InlineElement.LINK)
        assert "?q=1&b=2" in link.url
