"""Core module - main enums and state management.

This module provides the foundational types for termflow:

- Enums for parsing states (Code, ListType, BlockType, etc.)
- ParseState for tracking streaming parse state
- InlineState for inline formatting snapshots

Example:
    >>> from termflow.core import ParseState, Code, ListType
    >>> state = ParseState()
    >>> state.set_width(80)
    >>> state.enter_code_block(Code.BACKTICK, "python")
    >>> state.is_in_code()
    True
"""

from termflow.core.enums import (
    BlockType,
    Code,
    EmitFlag,
    ListBullet,
    ListType,
    OrderedBullet,
)
from termflow.core.state import InlineState, ParseState

__all__ = [
    "BlockType",
    "Code",
    "EmitFlag",
    "InlineState",
    "ListBullet",
    "ListType",
    "OrderedBullet",
    "ParseState",
]
