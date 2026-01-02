"""Core enums for termflow markdown parsing.

These enums represent the different states and types that can occur
during streaming markdown parsing. They're used by ParseState to
track the current parsing context.
"""

from dataclasses import dataclass
from enum import Enum, auto


class Code(Enum):
    """Type of code block or code-related state.

    Used to distinguish between different code block formats and
    special table parsing states.
    """

    SPACES = auto()  # Indented code block (4 spaces)
    BACKTICK = auto()  # Fenced code block (```)
    HEADER = auto()  # Table header row
    BODY = auto()  # Table body rows
    FLUSH = auto()  # Flush/reset state


class ListType(Enum):
    """Type of list being processed.

    Markdown supports two list types:
    - Bullet/unordered lists (*, -, +)
    - Ordered/numbered lists (1., 2., etc.)
    """

    BULLET = auto()  # Unordered list (*, -, +)
    ORDERED = auto()  # Ordered list (1., 2., etc.)


class BlockType(Enum):
    """Type of block-level element.

    Block elements change the rendering context and typically
    have special visual treatment (indentation, borders, etc.).
    """

    QUOTE = auto()  # Block quote (> prefix)
    THINK = auto()  # "Think" block (<think>...</think>) - for LLM chain-of-thought


class EmitFlag(Enum):
    """Flags for special emit behavior.

    Used to signal special rendering events to the output system.
    """

    HEADER1 = auto()  # H1 header event
    HEADER2 = auto()  # H2 header event
    FLUSH = auto()  # Force flush output


class ListBullet(Enum):
    """Unordered list bullet type.

    Different bullet characters can be used in markdown:
    - Dash (-)
    - Asterisk (*)
    - Plus (+)

    PLUS_EXPAND is a special case for expandable sections (+---).
    """

    DASH = auto()  # -
    ASTERISK = auto()  # *
    PLUS = auto()  # +
    PLUS_EXPAND = auto()  # +--- (expandable section marker)

    @classmethod
    def from_char(cls, char: str) -> "ListBullet | None":
        """Get ListBullet from character.

        Args:
            char: The bullet character (-, *, +)

        Returns:
            Corresponding ListBullet or None if invalid.

        Example:
            >>> ListBullet.from_char('-')
            <ListBullet.DASH: 1>
            >>> ListBullet.from_char('x')
            None
        """
        mapping = {
            "-": cls.DASH,
            "*": cls.ASTERISK,
            "+": cls.PLUS,
        }
        return mapping.get(char)

    def to_char(self) -> str:
        """Get the display character for this bullet.

        Returns:
            The bullet character for display.

        Example:
            >>> ListBullet.DASH.to_char()
            '•'
        """
        # We render all bullets as bullet points for consistency
        if self == ListBullet.PLUS_EXPAND:
            return "▶"  # Expandable marker
        return "•"  # Standard bullet


@dataclass(frozen=True, slots=True)
class OrderedBullet:
    """Ordered list bullet with number.

    Represents a numbered list item (1., 2., 3., etc.).

    Attributes:
        number: The list item number (1-based).

    Example:
        >>> bullet = OrderedBullet(1)
        >>> str(bullet)
        '1.'
        >>> bullet = OrderedBullet(42)
        >>> str(bullet)
        '42.'
    """

    number: int

    def __str__(self) -> str:
        """Return the string representation (e.g., '1.')."""
        return f"{self.number}."

    def width(self) -> int:
        """Calculate display width of this bullet.

        Returns:
            Width including the number and dot.

        Example:
            >>> OrderedBullet(1).width()
            2
            >>> OrderedBullet(100).width()
            4
        """
        return len(str(self))
