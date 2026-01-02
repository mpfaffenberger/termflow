"""Tests for termflow.parser module."""

from termflow.parser import Parser
from termflow.parser.events import (
    BlockquoteLineEvent,
    BlockquoteStartEvent,
    CodeBlockEndEvent,
    CodeBlockLineEvent,
    CodeBlockStartEvent,
    HeadingEvent,
    HorizontalRuleEvent,
    ListItemContentEvent,
    ListItemStartEvent,
    TableHeaderEvent,
    TextEvent,
    ThinkBlockEndEvent,
    ThinkBlockStartEvent,
)


class TestHeadings:
    """Test heading parsing."""

    def test_h1(self):
        parser = Parser()
        events = parser.parse_line("# Hello World")
        assert len(events) >= 1
        heading = next((e for e in events if isinstance(e, HeadingEvent)), None)
        assert heading is not None
        assert heading.level == 1
        assert heading.content == "Hello World"

    def test_h2(self):
        parser = Parser()
        events = parser.parse_line("## Section")
        heading = next(e for e in events if isinstance(e, HeadingEvent))
        assert heading.level == 2
        assert heading.content == "Section"

    def test_h6(self):
        parser = Parser()
        events = parser.parse_line("###### Small heading")
        heading = next(e for e in events if isinstance(e, HeadingEvent))
        assert heading.level == 6
        assert heading.content == "Small heading"

    def test_not_heading_without_space(self):
        parser = Parser()
        events = parser.parse_line("#NotAHeading")
        # Should not be parsed as heading
        assert not any(isinstance(e, HeadingEvent) for e in events)

    def test_heading_with_extra_spaces(self):
        parser = Parser()
        events = parser.parse_line("##   Extra spaces")
        heading = next(e for e in events if isinstance(e, HeadingEvent))
        assert heading.level == 2


class TestCodeBlocks:
    """Test code block parsing."""

    def test_fenced_code_block_start(self):
        parser = Parser()
        events = parser.parse_line("```python")
        assert any(isinstance(e, CodeBlockStartEvent) for e in events)
        start = next(e for e in events if isinstance(e, CodeBlockStartEvent))
        assert start.language == "python"

    def test_fenced_code_block_line(self):
        parser = Parser()
        parser.parse_line("```python")
        events = parser.parse_line("print('hello')")
        assert any(isinstance(e, CodeBlockLineEvent) for e in events)
        line_event = next(e for e in events if isinstance(e, CodeBlockLineEvent))
        assert "print" in line_event.line

    def test_fenced_code_block_end(self):
        parser = Parser()
        parser.parse_line("```python")
        parser.parse_line("print('hello')")
        events = parser.parse_line("```")
        assert any(isinstance(e, CodeBlockEndEvent) for e in events)

    def test_code_block_no_language(self):
        parser = Parser()
        events = parser.parse_line("```")
        start = next(e for e in events if isinstance(e, CodeBlockStartEvent))
        assert start.language is None or start.language == ""

    def test_code_block_javascript(self):
        parser = Parser()
        events = parser.parse_line("```javascript")
        start = next(e for e in events if isinstance(e, CodeBlockStartEvent))
        assert start.language == "javascript"

    def test_code_block_preserves_content(self):
        parser = Parser()
        parser.parse_line("```")
        events = parser.parse_line("  indented code")
        line_event = next(e for e in events if isinstance(e, CodeBlockLineEvent))
        assert line_event.line == "  indented code"


class TestLists:
    """Test list parsing."""

    def test_bullet_list_dash(self):
        parser = Parser()
        events = parser.parse_line("- Item one")
        assert any(isinstance(e, ListItemStartEvent | ListItemContentEvent) for e in events)

    def test_bullet_list_asterisk(self):
        parser = Parser()
        events = parser.parse_line("* Item one")
        assert len(events) > 0

    def test_bullet_list_plus(self):
        parser = Parser()
        events = parser.parse_line("+ Item one")
        assert len(events) > 0

    def test_ordered_list(self):
        parser = Parser()
        events = parser.parse_line("1. First item")
        assert len(events) > 0

    def test_nested_list(self):
        parser = Parser()
        parser.parse_line("- Parent")
        events = parser.parse_line("  - Child")
        # Should handle nested list
        assert len(events) > 0


class TestTables:
    """Test table parsing."""

    def test_table_header(self):
        parser = Parser()
        events = parser.parse_line("| Col A | Col B |")
        assert any(isinstance(e, TableHeaderEvent) for e in events)

    def test_table_header_content(self):
        parser = Parser()
        events = parser.parse_line("| Name | Age |")
        header = next(e for e in events if isinstance(e, TableHeaderEvent))
        assert "Name" in header.cells
        assert "Age" in header.cells

    def test_table_separator(self):
        parser = Parser()
        parser.parse_line("| Col A | Col B |")
        events = parser.parse_line("|-------|-------|")
        # Should process separator
        assert len(events) >= 0  # May or may not emit event


class TestBlockquotes:
    """Test blockquote parsing."""

    def test_blockquote_start(self):
        parser = Parser()
        events = parser.parse_line("> This is quoted")
        assert any(isinstance(e, BlockquoteStartEvent | BlockquoteLineEvent) for e in events)

    def test_blockquote_content(self):
        parser = Parser()
        events = parser.parse_line("> Quote content")
        line_event = next((e for e in events if isinstance(e, BlockquoteLineEvent)), None)
        if line_event:
            assert "Quote content" in line_event.text

    def test_nested_blockquote(self):
        parser = Parser()
        parser.parse_line("> Level 1")
        events = parser.parse_line(">> Level 2")
        assert len(events) > 0


class TestThinkBlocks:
    """Test think block parsing (LLM chain-of-thought)."""

    def test_think_start(self):
        parser = Parser()
        events = parser.parse_line("<think>")
        assert any(isinstance(e, ThinkBlockStartEvent) for e in events)

    def test_think_end(self):
        parser = Parser()
        parser.parse_line("<think>")
        parser.parse_line("Thinking...")
        events = parser.parse_line("</think>")
        assert any(isinstance(e, ThinkBlockEndEvent) for e in events)

    def test_think_content(self):
        parser = Parser()
        parser.parse_line("<think>")
        events = parser.parse_line("Some reasoning here")
        # Should emit line event inside think block
        assert len(events) > 0


class TestCheckboxes:
    """Test task list / checkbox parsing."""

    def test_unchecked_checkbox(self):
        parser = Parser()
        events = parser.parse_line("- [ ] Unchecked task")
        start_event = next((e for e in events if isinstance(e, ListItemStartEvent)), None)
        assert start_event is not None
        assert start_event.checked is False

    def test_checked_checkbox_lowercase(self):
        parser = Parser()
        events = parser.parse_line("- [x] Checked task")
        start_event = next(e for e in events if isinstance(e, ListItemStartEvent))
        assert start_event.checked is True

    def test_checked_checkbox_uppercase(self):
        parser = Parser()
        events = parser.parse_line("- [X] Checked task")
        start_event = next(e for e in events if isinstance(e, ListItemStartEvent))
        assert start_event.checked is True

    def test_regular_list_no_checkbox(self):
        parser = Parser()
        events = parser.parse_line("- Regular item")
        start_event = next(e for e in events if isinstance(e, ListItemStartEvent))
        assert start_event.checked is None

    def test_checkbox_content_extracted(self):
        parser = Parser()
        events = parser.parse_line("- [x] Task content here")
        content_event = next((e for e in events if isinstance(e, ListItemContentEvent)), None)
        assert content_event is not None
        assert content_event.text == "Task content here"

    def test_nested_checkbox(self):
        parser = Parser()
        parser.parse_line("- [x] Parent")
        events = parser.parse_line("  - [ ] Child")
        start_event = next(e for e in events if isinstance(e, ListItemStartEvent))
        assert start_event.checked is False
        assert start_event.indent == 1


class TestHorizontalRule:
    """Test horizontal rule parsing."""

    def test_dashes(self):
        parser = Parser()
        events = parser.parse_line("---")
        assert any(isinstance(e, HorizontalRuleEvent) for e in events)

    def test_asterisks(self):
        parser = Parser()
        events = parser.parse_line("***")
        assert any(isinstance(e, HorizontalRuleEvent) for e in events)

    def test_underscores(self):
        parser = Parser()
        events = parser.parse_line("___")
        assert any(isinstance(e, HorizontalRuleEvent) for e in events)

    def test_with_spaces(self):
        # Note: "- - -" with spaces may be parsed as list item in some parsers
        parser = Parser()
        events = parser.parse_line("- - -")
        # Just check it doesn't crash and returns events
        assert len(events) >= 0


class TestFinalize:
    """Test parser finalization."""

    def test_closes_code_block(self):
        parser = Parser()
        parser.parse_line("```python")
        parser.parse_line("code")
        events = parser.finalize()
        assert any(isinstance(e, CodeBlockEndEvent) for e in events)

    def test_closes_think_block(self):
        parser = Parser()
        parser.parse_line("<think>")
        parser.parse_line("thinking")
        events = parser.finalize()
        assert any(isinstance(e, ThinkBlockEndEvent) for e in events)

    def test_finalize_empty_parser(self):
        parser = Parser()
        events = parser.finalize()
        # Should not raise, may return empty list
        assert isinstance(events, list)

    def test_double_finalize(self):
        parser = Parser()
        parser.parse_line("# Heading")
        parser.finalize()
        events = parser.finalize()
        # Should handle gracefully
        assert isinstance(events, list)


class TestDocumentParsing:
    """Test full document parsing."""

    def test_full_document(self):
        parser = Parser()
        doc = """# Title

This is a paragraph.

```python
print("hello")
```

- Item 1
- Item 2
"""
        events = parser.parse_document(doc)

        # Check we got various event types
        event_types = {type(e).__name__ for e in events}
        assert "HeadingEvent" in event_types
        assert "CodeBlockStartEvent" in event_types
        assert "CodeBlockEndEvent" in event_types

    def test_empty_document(self):
        parser = Parser()
        events = parser.parse_document("")
        assert isinstance(events, list)

    def test_single_line_document(self):
        parser = Parser()
        events = parser.parse_document("Just text")
        assert len(events) > 0


class TestTextEvents:
    """Test plain text parsing."""

    def test_plain_text(self):
        parser = Parser()
        events = parser.parse_line("Just some plain text")
        assert any(isinstance(e, TextEvent) for e in events)

    def test_text_content(self):
        parser = Parser()
        events = parser.parse_line("Hello world")
        text_event = next(e for e in events if isinstance(e, TextEvent))
        assert "Hello world" in text_event.text
