"""Parser module - streaming markdown parsing.

This module provides the core parsing functionality for termflow:

- ParseEvent classes for representing parsed elements
- InlineParser for inline formatting (bold, italic, code, links, etc.)
- Parser for line-by-line streaming markdown parsing
- HTML entity decoding utilities

Example:
    >>> from termflow.parser import Parser, InlineParser
    >>> parser = Parser()
    >>> events = parser.parse_document("# Hello\n\nThis is **bold**!")
    >>> for event in events:
    ...     print(event)
"""

from termflow.parser.entities import (
    ENTITY_RE,
    decode_html_entities,
    decode_if_needed,
    has_html_entities,
)
from termflow.parser.events import (
    INLINE_EVENT_TYPES,
    BlockquoteEndEvent,
    BlockquoteLineEvent,
    BlockquoteStartEvent,
    BoldEvent,
    BoldItalicEvent,
    CodeBlockEndEvent,
    CodeBlockLineEvent,
    CodeBlockStartEvent,
    EmptyLineEvent,
    FootnoteEvent,
    HeadingEvent,
    HorizontalRuleEvent,
    ImageEvent,
    InlineCodeEvent,
    ItalicEvent,
    LinkEvent,
    ListEndEvent,
    ListItemContentEvent,
    ListItemEndEvent,
    ListItemStartEvent,
    NewlineEvent,
    ParagraphEndEvent,
    ParagraphStartEvent,
    ParseEvent,
    StrikeoutEvent,
    TableEndEvent,
    TableHeaderEvent,
    TableRowEvent,
    TableSeparatorEvent,
    TableStartEvent,
    TextEvent,
    ThinkBlockEndEvent,
    ThinkBlockLineEvent,
    ThinkBlockStartEvent,
    UnderlineEvent,
    is_block_event,
    is_inline_event,
)
from termflow.parser.inline import (
    InlineElement,
    InlineParser,
    InlineToken,
    strip_formatting,
    tokenize_inline,
)
from termflow.parser.parser import Parser, TableState

__all__ = [
    "ENTITY_RE",
    "INLINE_EVENT_TYPES",
    "BlockquoteEndEvent",
    "BlockquoteLineEvent",
    "BlockquoteStartEvent",
    "BoldEvent",
    "BoldItalicEvent",
    "CodeBlockEndEvent",
    "CodeBlockLineEvent",
    "CodeBlockStartEvent",
    "EmptyLineEvent",
    "FootnoteEvent",
    "HeadingEvent",
    "HorizontalRuleEvent",
    "ImageEvent",
    "InlineCodeEvent",
    "InlineElement",
    "InlineParser",
    "InlineToken",
    "ItalicEvent",
    "LinkEvent",
    "ListEndEvent",
    "ListItemContentEvent",
    "ListItemEndEvent",
    "ListItemStartEvent",
    "NewlineEvent",
    "ParagraphEndEvent",
    "ParagraphStartEvent",
    "ParseEvent",
    "Parser",
    "StrikeoutEvent",
    "TableEndEvent",
    "TableHeaderEvent",
    "TableRowEvent",
    "TableSeparatorEvent",
    "TableStartEvent",
    "TableState",
    "TextEvent",
    "ThinkBlockEndEvent",
    "ThinkBlockLineEvent",
    "ThinkBlockStartEvent",
    "UnderlineEvent",
    "decode_html_entities",
    "decode_if_needed",
    "has_html_entities",
    "is_block_event",
    "is_inline_event",
    "strip_formatting",
    "tokenize_inline",
]
