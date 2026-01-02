"""Streaming markdown parser.

This is the heart of termflow - a line-by-line parser designed for
streaming scenarios like LLM output. It maintains state across lines
and emits ParseEvent objects that can be consumed by renderers.

The parser handles:
- Fenced code blocks (```, ~~~, <pre>)
- Headings (# through ######)
- Lists (bullet and ordered, with nesting)
- Block quotes (>)
- Think blocks (<think>...</think>) for LLM chain-of-thought
- Tables (GitHub Flavored Markdown)
- Horizontal rules (---, ***, ___)
- Inline formatting (via InlineParser)
"""

from __future__ import annotations

import re
from enum import Enum, auto

from termflow.core import BlockType, Code, ListType, ParseState
from termflow.parser.events import (
    BlockquoteEndEvent,
    BlockquoteLineEvent,
    BlockquoteStartEvent,
    CodeBlockEndEvent,
    CodeBlockLineEvent,
    CodeBlockStartEvent,
    EmptyLineEvent,
    HeadingEvent,
    HorizontalRuleEvent,
    ListEndEvent,
    ListItemContentEvent,
    ListItemEndEvent,
    ListItemStartEvent,
    NewlineEvent,
    ParseEvent,
    TableEndEvent,
    TableHeaderEvent,
    TableRowEvent,
    TableSeparatorEvent,
    TableStartEvent,
    TextEvent,
    ThinkBlockEndEvent,
    ThinkBlockLineEvent,
    ThinkBlockStartEvent,
)
from termflow.parser.inline import InlineParser

# =============================================================================
# Regex Patterns (compiled once for performance)
# =============================================================================

# Code fence: ``` or ~~~ with optional language
CODE_FENCE_RE = re.compile(r"^(\s*)(```+|~~~+|<pre>)\s*([^\s`~]*).*$")
CODE_FENCE_END_RE = re.compile(r"^\s*(```+|~~~+|</pre>)\s*$")

# Headings: # through ######
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# ATX heading closing: ## Heading ## (optional closing hashes)
HEADING_CLOSE_RE = re.compile(r"\s+#+\s*$")

# List items: -, *, +, or 1. 2. etc.
LIST_ITEM_RE = re.compile(r"^(\s*)([+*-]|\d+\.)\s+(.*)$")

# Checkbox pattern: [ ] or [x] or [X] at start of list item content
CHECKBOX_RE = re.compile(r"^\[([ xX])\]\s+(.*)$")

# Expandable list item: +---
EXPAND_ITEM_RE = re.compile(r"^(\s*)\+-+\s*$")

# Block quote or think block
BLOCK_RE = re.compile(r"^\s*((>\s*)+|[<◁]\.?think[>▷]|</?\s*think\s*[>▷]?)(.*)$", re.IGNORECASE)

# Think block start/end patterns
THINK_START_RE = re.compile(r"^\s*(<think>|<\.think>|◁think▷|◁\.think▷)\s*$", re.IGNORECASE)
THINK_END_RE = re.compile(r"^\s*(</think>|</\.think>|◁/think▷|◁/\.think▷)\s*$", re.IGNORECASE)
THINK_INLINE_RE = re.compile(r"</?\s*think\s*>", re.IGNORECASE)

# Horizontal rule: ---, ***, ___
HR_RE = re.compile(r"^\s*([-]{3,}|[*]{3,}|[_]{3,})\s*$")

# Table row: | cell | cell |
TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")

# Table separator: |---|---| with optional alignment colons
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s|:\-]+\|?\s*$")
TABLE_SEP_CELL_RE = re.compile(r":?-+:?")

# Indented code block (4 spaces or 1 tab)
INDENT_CODE_RE = re.compile(r"^( {4}|\t)(.*)$")


class TableState(Enum):
    """State of table parsing."""

    HEADER = auto()
    SEPARATOR = auto()
    BODY = auto()


class Parser:
    """Streaming markdown parser.

    Designed to handle line-by-line input for streaming scenarios
    (like LLM output) while also working with complete documents.

    Example:
        >>> parser = Parser()
        >>> for line in markdown_stream:
        ...     events = parser.parse_line(line)
        ...     for event in events:
        ...         renderer.render(event)
        >>> events = parser.finalize()  # Close any open blocks
    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self.state = ParseState()
        self.inline_parser = InlineParser()
        self.code_fence: str | None = None  # The fence string (``` or ~~~)
        self.code_fence_indent: int = 0  # Indentation of opening fence
        self.table_state: TableState | None = None
        self.table_alignments: list[str] = []
        self._events: list[ParseEvent] = []
        self._prev_was_empty = False
        self._in_paragraph = False

    def parse_line(self, line: str) -> list[ParseEvent]:
        """Parse a single line and return events.

        Args:
            line: The line to parse (without trailing newline).

        Returns:
            List of ParseEvent objects generated from this line.
        """
        self._events.clear()

        # Strip trailing whitespace but preserve leading
        line = line.rstrip()

        # Handle code blocks first (they consume everything verbatim)
        if self.state.is_in_code():
            self._parse_in_code_block(line)
            return list(self._events)

        # Handle think blocks
        if self.state.block_type == BlockType.THINK:
            self._parse_in_think_block(line)
            return list(self._events)

        # Check for empty line
        if not line.strip():
            return self._handle_empty_line()

        # Track previous empty state
        was_prev_empty = self._prev_was_empty
        self._prev_was_empty = False
        self.state.last_line_empty = False

        # Try block constructs in order of precedence
        if self._try_parse_think_block(line):
            return list(self._events)
        if self._try_parse_code_fence(line):
            return list(self._events)
        if self._try_parse_heading(line):
            return list(self._events)
        if self._try_parse_hr(line):
            return list(self._events)
        if self._try_parse_block_quote(line):
            return list(self._events)
        if self._try_parse_list_item(line):
            return list(self._events)
        if self._try_parse_table(line):
            return list(self._events)

        # Check for indented code block (only after empty line)
        if was_prev_empty and self._try_parse_indented_code(line):
            return list(self._events)

        # Exit special contexts for plain text
        self._exit_block_contexts()

        # Parse as inline content (paragraph text)
        self._parse_inline_content(line)
        return list(self._events)

    def parse_document(self, content: str) -> list[ParseEvent]:
        """Parse a complete document.

        Args:
            content: The full markdown document.

        Returns:
            List of all ParseEvent objects.
        """
        all_events: list[ParseEvent] = []
        for line in content.splitlines():
            all_events.extend(self.parse_line(line))
        all_events.extend(self.finalize())
        return all_events

    def finalize(self) -> list[ParseEvent]:
        """Finalize parsing, closing any open blocks.

        Call this after all input has been processed to properly
        close any open code blocks, lists, tables, etc.

        Returns:
            List of closing events.
        """
        self._events.clear()

        # Close code block
        if self.state.is_in_code():
            self._events.append(CodeBlockEndEvent())
            self.state.exit_code_block()
            self.code_fence = None

        # Close think block
        if self.state.block_type == BlockType.THINK:
            self._events.append(ThinkBlockEndEvent())
            self.state.exit_block()

        # Close block quotes
        if self.state.block_depth > 0:
            self._events.append(BlockquoteEndEvent())
            while self.state.block_depth > 0:
                self.state.exit_block()

        # Close lists
        if self.state.in_list:
            self._exit_list_context()

        # Close table
        if self.table_state is not None:
            self._events.append(TableEndEvent())
            self.table_state = None
            self.state.in_table = None

        return list(self._events)

    def reset(self) -> None:
        """Reset parser to initial state."""
        self.state = ParseState()
        self.inline_parser = InlineParser()
        self.code_fence = None
        self.code_fence_indent = 0
        self.table_state = None
        self.table_alignments.clear()
        self._events.clear()
        self._prev_was_empty = False
        self._in_paragraph = False

    # =========================================================================
    # Code Block Handling
    # =========================================================================

    def _parse_in_code_block(self, line: str) -> None:
        """Parse a line inside a code block."""
        # Check for closing fence
        if self.code_fence:
            match = CODE_FENCE_END_RE.match(line)
            if match:
                fence = match.group(1)
                # Must match opening fence type and be at least as long
                if (
                    fence.startswith(self.code_fence[0]) and len(fence) >= len(self.code_fence)
                ) or (self.code_fence == "<pre>" and fence == "</pre>"):
                    self._events.append(CodeBlockEndEvent())
                    self.state.exit_code_block()
                    self.code_fence = None
                    return

        # Remove indentation matching the fence
        if self.code_fence_indent > 0 and line.startswith(" " * self.code_fence_indent):
            line = line[self.code_fence_indent :]

        self._events.append(CodeBlockLineEvent(line=line))

    def _try_parse_code_fence(self, line: str) -> bool:
        """Try to parse a code fence start."""
        match = CODE_FENCE_RE.match(line)
        if not match:
            return False

        indent = len(match.group(1))
        fence = match.group(2)
        language = match.group(3).strip() or None

        # Exit other contexts
        self._exit_block_contexts()

        self.code_fence = fence
        self.code_fence_indent = indent
        self.state.enter_code_block(Code.BACKTICK, language)
        self._events.append(CodeBlockStartEvent(language=language, indent=indent))
        return True

    def _try_parse_indented_code(self, line: str) -> bool:
        """Try to parse an indented code block (4 spaces)."""
        match = INDENT_CODE_RE.match(line)
        if not match:
            return False

        # Only start indented code after empty line and not in list
        if self.state.in_list:
            return False

        code_line = match.group(2)

        if not self.state.is_in_code():
            self.state.enter_code_block(Code.SPACES)
            self._events.append(CodeBlockStartEvent(language=None, indent=4))

        self._events.append(CodeBlockLineEvent(line=code_line))
        return True

    # =========================================================================
    # Think Block Handling
    # =========================================================================

    def _try_parse_think_block(self, line: str) -> bool:
        """Try to parse a think block start."""
        if THINK_START_RE.match(line):
            self._exit_block_contexts()
            self.state.enter_block(BlockType.THINK)
            self._events.append(ThinkBlockStartEvent())
            return True
        return False

    def _parse_in_think_block(self, line: str) -> None:
        """Parse a line inside a think block."""
        # Check for closing tag
        if THINK_END_RE.match(line):
            self._events.append(ThinkBlockEndEvent())
            self.state.exit_block()
            return

        # Remove inline think tags if present
        line = THINK_INLINE_RE.sub("", line)
        self._events.append(ThinkBlockLineEvent(text=line))

    # =========================================================================
    # Heading Handling
    # =========================================================================

    def _try_parse_heading(self, line: str) -> bool:
        """Try to parse a heading."""
        match = HEADING_RE.match(line)
        if not match:
            return False

        level = len(match.group(1))
        content = match.group(2)

        # Remove trailing hashes if present (ATX closing style)
        content = HEADING_CLOSE_RE.sub("", content).strip()

        # Exit other contexts
        self._exit_block_contexts()

        self._events.append(HeadingEvent(level=level, content=content))
        return True

    # =========================================================================
    # Horizontal Rule Handling
    # =========================================================================

    def _try_parse_hr(self, line: str) -> bool:
        """Try to parse a horizontal rule."""
        if HR_RE.match(line):
            self._exit_block_contexts()
            self._events.append(HorizontalRuleEvent())
            return True
        return False

    # =========================================================================
    # Block Quote Handling
    # =========================================================================

    def _try_parse_block_quote(self, line: str) -> bool:
        """Try to parse a block quote line."""
        # Count leading > characters
        stripped = line.lstrip()
        if not stripped.startswith(">"):
            return False

        # Count depth and extract content
        depth = 0
        content = stripped
        while content.startswith(">"):
            depth += 1
            content = content[1:].lstrip()
            # Handle > without space
            if content.startswith(" "):
                content = content[1:]

        # Adjust block depth
        while self.state.block_depth < depth:
            self.state.enter_block(BlockType.QUOTE)
            self._events.append(BlockquoteStartEvent(depth=self.state.block_depth))

        while self.state.block_depth > depth:
            self.state.exit_block()
            self._events.append(BlockquoteEndEvent())

        self._events.append(BlockquoteLineEvent(text=content, depth=depth))
        return True

    # =========================================================================
    # List Handling
    # =========================================================================

    def _try_parse_list_item(self, line: str) -> bool:
        """Try to parse a list item."""
        match = LIST_ITEM_RE.match(line)
        if not match:
            # Check for expandable item (+---)
            expand_match = EXPAND_ITEM_RE.match(line)
            if expand_match:
                indent = len(expand_match.group(1))
                self._handle_list_item(indent, "-", "", is_expand=True)
                return True
            return False

        indent = len(match.group(1))
        bullet = match.group(2)
        content = match.group(3)

        self._handle_list_item(indent, bullet, content)
        return True

    def _handle_list_item(
        self, indent: int, bullet: str, content: str, is_expand: bool = False
    ) -> None:
        """Handle a parsed list item."""
        # Close table if open
        if self.table_state is not None:
            self._events.append(TableEndEvent())
            self.table_state = None
            self.state.in_table = None

        # Determine list type
        is_ordered = bullet.endswith(".")
        list_type = ListType.ORDERED if is_ordered else ListType.BULLET
        number: int | None = None

        if is_ordered:
            try:
                number = int(bullet[:-1])
            except ValueError:
                number = 1

        # Handle nesting based on indentation
        current_indent = self.state.current_list_indent()

        if not self.state.in_list:
            # Start new list
            self.state.push_list(indent, list_type)
        elif indent > current_indent:
            # Nested list
            self.state.push_list(indent, list_type)
        elif indent < current_indent:
            # Dedent - pop lists until we find matching indent
            while self.state.list_depth() > 0 and self.state.current_list_indent() > indent:
                self.state.pop_list()
                self._events.append(ListItemEndEvent())
                self._events.append(ListEndEvent())
            # Check if we need to start a new list at this level
            if not self.state.in_list or self.state.current_list_type() != list_type:
                if self.state.in_list:
                    self.state.pop_list()
                    self._events.append(ListItemEndEvent())
                    self._events.append(ListEndEvent())
                self.state.push_list(indent, list_type)

        # Get ordered list number if applicable
        if is_ordered and not is_expand:
            number = self.state.next_list_number()

        # Check for checkbox pattern in content
        checked: bool | None = None
        actual_content = content
        if content and not is_expand:
            checkbox_match = CHECKBOX_RE.match(content)
            if checkbox_match:
                check_char = checkbox_match.group(1)
                checked = check_char.lower() == "x"
                actual_content = checkbox_match.group(2)

        # Emit list item events
        bullet_char = "▶" if is_expand else ("•" if not is_ordered else None)
        self._events.append(
            ListItemStartEvent(
                indent=self.state.list_depth() - 1,
                ordered=is_ordered,
                number=number,
                bullet_char=bullet_char or "•",
                checked=checked,
            )
        )
        if actual_content:
            self._events.append(ListItemContentEvent(text=actual_content))

    def _exit_list_context(self) -> None:
        """Exit all list levels."""
        while self.state.list_depth() > 0:
            self.state.pop_list()
            self._events.append(ListItemEndEvent())
            self._events.append(ListEndEvent())

    # =========================================================================
    # Table Handling
    # =========================================================================

    def _try_parse_table(self, line: str) -> bool:
        """Try to parse a table row."""
        row_match = TABLE_ROW_RE.match(line)

        if row_match:
            cells = self._parse_table_cells(row_match.group(1))

            if self.table_state is None:
                # Start new table - this is the header row
                self._exit_block_contexts()
                self._events.append(TableStartEvent())
                self.state.enter_table(Code.HEADER)
                self.table_state = TableState.HEADER
                self._events.append(TableHeaderEvent(cells=tuple(cells)))
            elif self.table_state == TableState.HEADER:
                # Could be separator or another header row
                # Check if it's a separator
                if self._is_table_separator(row_match.group(1)):
                    self.table_state = TableState.SEPARATOR
                    self._parse_table_alignments(row_match.group(1))
                    self._events.append(
                        TableSeparatorEvent(alignments=tuple(self.table_alignments))
                    )
                else:
                    # Another header row
                    self._events.append(TableHeaderEvent(cells=tuple(cells)))
            elif self.table_state == TableState.SEPARATOR:
                # First body row after separator
                self.table_state = TableState.BODY
                self.state.in_table = Code.BODY
                self._events.append(TableRowEvent(cells=tuple(cells)))
            else:
                # Body row
                self._events.append(TableRowEvent(cells=tuple(cells)))

            return True

        # Check for separator without pipes on edges
        if self.table_state == TableState.HEADER and TABLE_SEP_RE.match(line):
            self.table_state = TableState.SEPARATOR
            self._parse_table_alignments(line)
            self._events.append(TableSeparatorEvent(alignments=tuple(self.table_alignments)))
            return True

        # If we were in a table but this line doesn't match, end the table
        if self.table_state is not None:
            self._events.append(TableEndEvent())
            self.table_state = None
            self.state.in_table = None
            # Don't return True - let the line be parsed as something else

        return False

    def _parse_table_cells(self, content: str) -> list[str]:
        """Parse table cells from row content."""
        cells = content.split("|")
        return [cell.strip() for cell in cells]

    def _is_table_separator(self, content: str) -> bool:
        """Check if content is a table separator row."""
        cells = content.split("|")
        for cell in cells:
            cell = cell.strip()
            if cell and not TABLE_SEP_CELL_RE.fullmatch(cell):
                return False
        return any(cells)  # Must have at least one cell

    def _parse_table_alignments(self, content: str) -> None:
        """Parse column alignments from separator row."""
        self.table_alignments.clear()
        cells = content.split("|")
        for cell in cells:
            cell = cell.strip()
            if not cell or cell == "-" * len(cell):
                continue
            if cell.startswith(":") and cell.endswith(":"):
                self.table_alignments.append("center")
            elif cell.endswith(":"):
                self.table_alignments.append("right")
            elif cell.startswith(":"):
                self.table_alignments.append("left")
            else:
                self.table_alignments.append("none")

    # =========================================================================
    # Empty Line Handling
    # =========================================================================

    def _handle_empty_line(self) -> list[ParseEvent]:
        """Handle an empty line."""
        self._prev_was_empty = True
        self.state.last_line_empty = True

        # Close indented code block
        if self.state.in_code == Code.SPACES:
            self._events.append(CodeBlockEndEvent())
            self.state.exit_code_block()

        # Close table
        if self.table_state is not None:
            self._events.append(TableEndEvent())
            self.table_state = None
            self.state.in_table = None

        # Don't close block quotes or lists on single empty line
        # (allows for multi-paragraph list items and quotes)

        self._events.append(EmptyLineEvent())
        return list(self._events)

    # =========================================================================
    # Context Management
    # =========================================================================

    def _exit_block_contexts(self) -> None:
        """Exit block contexts when switching to different content."""
        # Close block quotes on non-quote content
        if self.state.block_depth > 0 and self.state.block_type == BlockType.QUOTE:
            self._events.append(BlockquoteEndEvent())
            while self.state.block_depth > 0:
                self.state.exit_block()

        # Close lists on non-list content (with empty line before)
        if self.state.in_list and self._prev_was_empty:
            self._exit_list_context()

        # Close table
        if self.table_state is not None:
            self._events.append(TableEndEvent())
            self.table_state = None
            self.state.in_table = None

    # =========================================================================
    # Inline Content
    # =========================================================================

    def _parse_inline_content(self, line: str) -> None:
        """Parse a line as inline content."""
        # For now, emit as simple text
        # The renderer will use InlineParser to handle formatting
        self._events.append(TextEvent(text=line))
        self._events.append(NewlineEvent())
