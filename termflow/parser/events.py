"""Parse events for the streaming markdown parser.

Each event type represents a parsed markdown element. The parser emits
these events as it processes input, and renderers consume them to
produce formatted output.

Events are organized into:
- Inline events (text, code, bold, italic, links, etc.)
- Block events (headings, code blocks, lists, tables, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# =============================================================================
# Inline Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class TextEvent:
    """Plain text content."""

    text: str


@dataclass(frozen=True, slots=True)
class InlineCodeEvent:
    """Inline code span (`code`)."""

    code: str


@dataclass(frozen=True, slots=True)
class BoldEvent:
    """Bold text (**bold**)."""

    text: str


@dataclass(frozen=True, slots=True)
class ItalicEvent:
    """Italic text (*italic*)."""

    text: str


@dataclass(frozen=True, slots=True)
class BoldItalicEvent:
    """Bold and italic text (***both***)."""

    text: str


@dataclass(frozen=True, slots=True)
class UnderlineEvent:
    """Underlined text."""

    text: str


@dataclass(frozen=True, slots=True)
class StrikeoutEvent:
    """Strikethrough text (~~strike~~)."""

    text: str


@dataclass(frozen=True, slots=True)
class LinkEvent:
    """A hyperlink ([text](url))."""

    text: str
    url: str


@dataclass(frozen=True, slots=True)
class ImageEvent:
    """An image reference (![alt](url))."""

    alt: str
    url: str


@dataclass(frozen=True, slots=True)
class FootnoteEvent:
    """Footnote reference ([^N])."""

    number: str


# =============================================================================
# Block Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class HeadingEvent:
    """Heading (H1-H6)."""

    level: int  # 1-6
    content: str


@dataclass(frozen=True, slots=True)
class CodeBlockStartEvent:
    """Start of a fenced code block."""

    language: str | None
    indent: int = 0


@dataclass(frozen=True, slots=True)
class CodeBlockLineEvent:
    """A line of code within a code block."""

    line: str


@dataclass(frozen=True, slots=True)
class CodeBlockEndEvent:
    """End of a code block."""

    pass


@dataclass(frozen=True, slots=True)
class ListItemStartEvent:
    """Start of a list item."""

    indent: int
    ordered: bool
    number: int | None = None  # For ordered lists
    bullet_char: str = "•"  # Display character
    checked: bool | None = None  # For task lists: True=checked, False=unchecked, None=not a task


@dataclass(frozen=True, slots=True)
class ListItemContentEvent:
    """Content within a list item (may contain inline formatting)."""

    text: str


@dataclass(frozen=True, slots=True)
class ListItemEndEvent:
    """End of a list item."""

    pass


@dataclass(frozen=True, slots=True)
class ListEndEvent:
    """End of a list."""

    pass


@dataclass(frozen=True, slots=True)
class TableStartEvent:
    """Start of a table."""

    pass


@dataclass(frozen=True, slots=True)
class TableHeaderEvent:
    """Table header row."""

    cells: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class TableRowEvent:
    """Table body row."""

    cells: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class TableSeparatorEvent:
    """Table separator line (|---|---|)."""

    alignments: tuple[str, ...] = field(default_factory=tuple)  # 'left', 'center', 'right', 'none'


@dataclass(frozen=True, slots=True)
class TableEndEvent:
    """End of a table."""

    pass


@dataclass(frozen=True, slots=True)
class BlockquoteStartEvent:
    """Start of a blockquote."""

    depth: int = 1


@dataclass(frozen=True, slots=True)
class BlockquoteLineEvent:
    """A line within a blockquote."""

    text: str
    depth: int = 1


@dataclass(frozen=True, slots=True)
class BlockquoteEndEvent:
    """End of a blockquote."""

    pass


@dataclass(frozen=True, slots=True)
class ThinkBlockStartEvent:
    """Start of a think block (<think>)."""

    pass


@dataclass(frozen=True, slots=True)
class ThinkBlockLineEvent:
    """A line within a think block."""

    text: str


@dataclass(frozen=True, slots=True)
class ThinkBlockEndEvent:
    """End of a think block (</think>)."""

    pass


@dataclass(frozen=True, slots=True)
class HorizontalRuleEvent:
    """Horizontal rule (---, ***, ___)."""

    pass


@dataclass(frozen=True, slots=True)
class EmptyLineEvent:
    """An empty/blank line."""

    pass


@dataclass(frozen=True, slots=True)
class NewlineEvent:
    """End of a line (for inline content)."""

    pass


@dataclass(frozen=True, slots=True)
class ParagraphStartEvent:
    """Start of a paragraph."""

    pass


@dataclass(frozen=True, slots=True)
class ParagraphEndEvent:
    """End of a paragraph."""

    pass


# =============================================================================
# Type Alias and Helpers
# =============================================================================

# Type alias for all event types
# Using Union for compatibility with older typing tools
ParseEvent = (
    TextEvent
    | InlineCodeEvent
    | BoldEvent
    | ItalicEvent
    | BoldItalicEvent
    | UnderlineEvent
    | StrikeoutEvent
    | LinkEvent
    | ImageEvent
    | FootnoteEvent
    | HeadingEvent
    | CodeBlockStartEvent
    | CodeBlockLineEvent
    | CodeBlockEndEvent
    | ListItemStartEvent
    | ListItemContentEvent
    | ListItemEndEvent
    | ListEndEvent
    | TableStartEvent
    | TableHeaderEvent
    | TableRowEvent
    | TableSeparatorEvent
    | TableEndEvent
    | BlockquoteStartEvent
    | BlockquoteLineEvent
    | BlockquoteEndEvent
    | ThinkBlockStartEvent
    | ThinkBlockLineEvent
    | ThinkBlockEndEvent
    | HorizontalRuleEvent
    | EmptyLineEvent
    | NewlineEvent
    | ParagraphStartEvent
    | ParagraphEndEvent
)

# Tuple of inline event types for isinstance checks
INLINE_EVENT_TYPES = (
    TextEvent,
    InlineCodeEvent,
    BoldEvent,
    ItalicEvent,
    BoldItalicEvent,
    UnderlineEvent,
    StrikeoutEvent,
    LinkEvent,
    ImageEvent,
    FootnoteEvent,
)


def is_inline_event(event: ParseEvent) -> bool:
    """Check if an event is an inline element.

    Args:
        event: Any parse event.

    Returns:
        True if the event represents inline content.
    """
    return isinstance(event, INLINE_EVENT_TYPES)


def is_block_event(event: ParseEvent) -> bool:
    """Check if an event is a block-level element.

    Args:
        event: Any parse event.

    Returns:
        True if the event represents block-level content.
    """
    return not isinstance(event, INLINE_EVENT_TYPES)
