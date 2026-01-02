"""Parse state management for streaming markdown processing.

This module contains the core state classes that track everything needed
to incrementally parse markdown as it streams in character by character.

The ParseState class is the heart of termflow - it maintains:
- Input buffer state for incremental parsing
- Width/layout configuration
- Code block state (language, buffer, indentation)
- List stack (nested lists with bullet types)
- Inline formatting flags (bold, italic, code, etc.)
- Block quote depth and type
- Feature flags for optional behaviors
"""

from dataclasses import dataclass, field

from termflow.core.enums import BlockType, Code, ListType


@dataclass(slots=True)
class InlineState:
    """Snapshot of current inline formatting states.

    This is a lightweight snapshot of the inline formatting flags
    from ParseState, useful for saving/restoring state or comparing
    formatting across positions.

    Attributes:
        inline_code: Inside inline code (`code`).
        in_bold: Inside bold text (**bold**).
        in_italic: Inside italic text (*italic* or _italic_).
        in_underline: Inside underlined text.
        in_strikeout: Inside strikethrough text (~~strike~~).
    """

    inline_code: bool = False
    in_bold: bool = False
    in_italic: bool = False
    in_underline: bool = False
    in_strikeout: bool = False

    def is_active(self) -> bool:
        """Check if any inline formatting is active.

        Returns:
            True if any formatting flag is set.
        """
        return any(
            [
                self.inline_code,
                self.in_bold,
                self.in_italic,
                self.in_underline,
                self.in_strikeout,
            ]
        )

    def __bool__(self) -> bool:
        """Boolean conversion - True if any formatting active."""
        return self.is_active()


@dataclass
class ParseState:
    """Main parse state for streaming markdown processing.

    Maintains all state needed to incrementally parse markdown as it streams in.
    This is the heart of termflow's streaming capability - we can pause parsing
    at any point and resume later with the same state.

    The state is organized into logical groups:
    - Input buffer state
    - Width/layout configuration
    - Indentation tracking
    - Code block state
    - List state (with nesting)
    - Inline formatting flags
    - Block state (quotes, think blocks)
    - Feature flags

    Example:
        >>> state = ParseState()
        >>> state.set_width(80)
        >>> state.enter_code_block(Code.BACKTICK, "python")
        >>> state.is_in_code()
        True
    """

    # =========================================================================
    # Input Buffer State
    # =========================================================================
    buffer: bytes = field(default_factory=bytes)
    current_line: str = ""
    first_line: bool = True
    last_line_empty: bool = True

    # =========================================================================
    # Width Configuration
    # =========================================================================
    width_arg: int | None = None  # User-specified width override
    width_full: int | None = None  # Detected terminal width
    width_wrap: bool = False  # Whether to wrap text

    # =========================================================================
    # Indentation State
    # =========================================================================
    first_indent: int | None = None  # First line indentation
    has_newline: bool = False  # Whether we've seen a newline
    bg: str = "\x1b[49m"  # Background color (BGRESET)

    # =========================================================================
    # Code Block State
    # =========================================================================
    code_buffer: str = ""  # Accumulated code content
    code_buffer_raw: str = ""  # Raw code before processing
    code_gen: int = 0  # Code block generation counter
    code_language: str | None = None  # Language for syntax highlighting
    code_first_line: bool = False  # Is this the first line of code block?
    code_indent: int = 0  # Indentation level for code

    # =========================================================================
    # List State
    # =========================================================================
    ordered_list_numbers: list[int] = field(default_factory=list)
    list_item_stack: list[tuple[int, ListType]] = field(default_factory=list)
    list_indent_text: int = 0  # Current list indentation in chars

    # =========================================================================
    # Block/Inline State Flags
    # =========================================================================
    in_list: bool = False
    in_code: Code | None = None
    inline_code: bool = False
    in_bold: bool = False
    in_italic: bool = False
    in_table: Code | None = None
    in_underline: bool = False
    in_strikeout: bool = False
    block_depth: int = 0
    block_type: BlockType | None = None

    # =========================================================================
    # Feature Flags
    # =========================================================================
    links: bool = True  # Render links
    images: bool = True  # Render images (as text)
    code_spaces: bool = False  # Allow space-indented code blocks
    clipboard: bool = True  # Enable clipboard operations
    savebrace: bool = True  # Save brace content

    # =========================================================================
    # Inline State Operations
    # =========================================================================

    def current(self) -> InlineState:
        """Returns snapshot of current inline formatting states.

        Returns:
            InlineState with current formatting flags.

        Example:
            >>> state = ParseState()
            >>> state.in_bold = True
            >>> snapshot = state.current()
            >>> snapshot.in_bold
            True
        """
        return InlineState(
            inline_code=self.inline_code,
            in_bold=self.in_bold,
            in_italic=self.in_italic,
            in_underline=self.in_underline,
            in_strikeout=self.in_strikeout,
        )

    def reset_inline(self) -> None:
        """Reset all inline formatting states.

        Call this when exiting a block that should clear formatting,
        like finishing a paragraph or entering a code block.
        """
        self.inline_code = False
        self.in_bold = False
        self.in_italic = False
        self.in_underline = False
        self.in_strikeout = False

    def has_inline_formatting(self) -> bool:
        """Check if any inline formatting is active.

        Returns:
            True if any inline formatting flag is set.
        """
        return any(
            [
                self.inline_code,
                self.in_bold,
                self.in_italic,
                self.in_underline,
                self.in_strikeout,
            ]
        )

    # =========================================================================
    # Width Operations
    # =========================================================================

    def set_width(self, width: int) -> None:
        """Set the terminal width.

        Args:
            width: Terminal width in columns.
        """
        self.width_full = width

    def full_width(self, offset: int = 0) -> int:
        """Calculate full width with optional offset.

        Args:
            offset: Amount to subtract from width.

        Returns:
            Available width, minimum 0.
        """
        base = self.width_full or 80
        return max(0, base - offset)

    def current_width(self, listwidth: bool = False) -> int:
        """Calculate current usable width for content.

        Takes into account block quote depth and optionally list indentation.

        Args:
            listwidth: Whether to account for list indentation.

        Returns:
            Available width for content.
        """
        base = self.width_full or 80
        block_offset = self.block_depth * 2  # "│ " per block level
        list_offset = self.list_indent_text if listwidth else 0
        return max(0, base - block_offset - list_offset)

    def space_left(self, listwidth: bool = False) -> str:
        """Generate left margin string for current context.

        Creates the visual prefix for the current nesting level,
        including block quote bars and list indentation.

        Args:
            listwidth: Whether to include list indentation.

        Returns:
            String to prefix content with.

        Example:
            >>> state = ParseState()
            >>> state.block_depth = 2
            >>> state.space_left()
            '│ │ '
        """
        result = "│ " * self.block_depth
        if listwidth and self.list_indent_text > 0:
            result += " " * self.list_indent_text
        return result

    # =========================================================================
    # Code Block Operations
    # =========================================================================

    def is_in_code(self) -> bool:
        """Check if currently inside any code block.

        Returns:
            True if in a code block (fenced or indented).
        """
        return self.in_code is not None

    def enter_code_block(self, code_type: Code, language: str | None = None) -> None:
        """Enter a code block.

        Args:
            code_type: Type of code block (SPACES or BACKTICK).
            language: Optional language for syntax highlighting.
        """
        self.in_code = code_type
        self.code_language = language
        self.code_first_line = True
        self.code_buffer = ""
        self.code_buffer_raw = ""
        self.code_gen += 1

    def exit_code_block(self) -> None:
        """Exit the current code block."""
        self.in_code = None
        self.code_language = None
        self.code_first_line = False

    def append_code(self, text: str) -> None:
        """Append text to the code buffer.

        Args:
            text: Code text to append.
        """
        self.code_buffer += text
        self.code_buffer_raw += text

    def clear_code_buffer(self) -> str:
        """Clear and return the code buffer.

        Returns:
            The accumulated code content.
        """
        content = self.code_buffer
        self.code_buffer = ""
        return content

    # =========================================================================
    # Table Operations
    # =========================================================================

    def is_in_table(self) -> bool:
        """Check if currently inside a table.

        Returns:
            True if parsing a table.
        """
        return self.in_table is not None

    def enter_table(self, table_type: Code = Code.HEADER) -> None:
        """Enter a table.

        Args:
            table_type: Initial table state (usually HEADER).
        """
        self.in_table = table_type

    def exit_table(self) -> None:
        """Exit the current table."""
        self.in_table = None

    # =========================================================================
    # List Operations
    # =========================================================================

    def push_list(self, indent: int, list_type: ListType) -> None:
        """Push a new list level onto the stack.

        Args:
            indent: Character indentation for this list level.
            list_type: Whether this is a bullet or ordered list.
        """
        self.list_item_stack.append((indent, list_type))
        if list_type == ListType.ORDERED:
            self.ordered_list_numbers.append(1)
        self.in_list = True

    def pop_list(self) -> tuple[int, ListType] | None:
        """Pop current list level from stack.

        Returns:
            The popped (indent, list_type) tuple, or None if stack empty.
        """
        if not self.list_item_stack:
            return None
        result = self.list_item_stack.pop()
        if result[1] == ListType.ORDERED and self.ordered_list_numbers:
            self.ordered_list_numbers.pop()
        self.in_list = bool(self.list_item_stack)
        return result

    def list_depth(self) -> int:
        """Get current list nesting depth.

        Returns:
            Number of nested list levels (0 if not in list).
        """
        return len(self.list_item_stack)

    def current_list_type(self) -> ListType | None:
        """Get the current list type.

        Returns:
            Current list type or None if not in a list.
        """
        if not self.list_item_stack:
            return None
        return self.list_item_stack[-1][1]

    def current_list_indent(self) -> int:
        """Get the current list indentation.

        Returns:
            Current list indentation or 0 if not in a list.
        """
        if not self.list_item_stack:
            return 0
        return self.list_item_stack[-1][0]

    def next_list_number(self) -> int | None:
        """Get and increment current ordered list number.

        Returns:
            The current number (before increment), or None if not in ordered list.
        """
        if not self.ordered_list_numbers:
            return None
        current = self.ordered_list_numbers[-1]
        self.ordered_list_numbers[-1] += 1
        return current

    def peek_list_number(self) -> int | None:
        """Peek at current ordered list number without incrementing.

        Returns:
            The current number, or None if not in ordered list.
        """
        if not self.ordered_list_numbers:
            return None
        return self.ordered_list_numbers[-1]

    def reset_list_number(self) -> None:
        """Reset the current ordered list number to 1."""
        if self.ordered_list_numbers:
            self.ordered_list_numbers[-1] = 1

    def clear_lists(self) -> None:
        """Clear all list state."""
        self.list_item_stack.clear()
        self.ordered_list_numbers.clear()
        self.list_indent_text = 0
        self.in_list = False

    # =========================================================================
    # Block Operations
    # =========================================================================

    def enter_block(self, block_type: BlockType) -> None:
        """Enter a block quote or think block.

        Args:
            block_type: Type of block to enter.
        """
        self.block_depth += 1
        self.block_type = block_type

    def exit_block(self) -> None:
        """Exit one level of block quote."""
        if self.block_depth > 0:
            self.block_depth -= 1
        if self.block_depth == 0:
            self.block_type = None

    def is_in_block(self) -> bool:
        """Check if inside a block quote or think block.

        Returns:
            True if block_depth > 0.
        """
        return self.block_depth > 0

    def is_in_quote(self) -> bool:
        """Check if inside a block quote specifically.

        Returns:
            True if in a QUOTE block.
        """
        return self.block_type == BlockType.QUOTE and self.block_depth > 0

    def is_in_think(self) -> bool:
        """Check if inside a think block specifically.

        Returns:
            True if in a THINK block.
        """
        return self.block_type == BlockType.THINK and self.block_depth > 0

    # =========================================================================
    # Buffer Operations
    # =========================================================================

    def append_buffer(self, data: bytes) -> None:
        """Append data to the input buffer.

        Args:
            data: Bytes to append.
        """
        self.buffer += data

    def clear_buffer(self) -> bytes:
        """Clear and return the input buffer.

        Returns:
            The buffer contents.
        """
        content = self.buffer
        self.buffer = b""
        return content

    def set_line(self, line: str) -> None:
        """Set the current line being processed.

        Args:
            line: The line text.
        """
        self.current_line = line
        if self.first_line:
            self.first_line = False

    # =========================================================================
    # Reset Operations
    # =========================================================================

    def reset(self) -> None:
        """Reset all state to initial values.

        Use this when starting to parse a new document.
        """
        # Buffer state
        self.buffer = b""
        self.current_line = ""
        self.first_line = True
        self.last_line_empty = True

        # Indentation
        self.first_indent = None
        self.has_newline = False

        # Code block
        self.code_buffer = ""
        self.code_buffer_raw = ""
        self.code_language = None
        self.code_first_line = False
        self.code_indent = 0
        self.in_code = None

        # Lists
        self.clear_lists()

        # Inline formatting
        self.reset_inline()

        # Table
        self.in_table = None

        # Blocks
        self.block_depth = 0
        self.block_type = None
