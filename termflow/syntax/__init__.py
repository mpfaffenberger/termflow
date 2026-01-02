"""Syntax highlighting module for code blocks.

This module provides Pygments-based syntax highlighting with:
- Language alias mapping (py → python, js → javascript, etc.)
- True-color (24-bit) and 256-color terminal support
- Lexer caching for performance
- Streaming-friendly line-by-line highlighting

Example:
    >>> from termflow.syntax import Highlighter, highlight_code
    >>> # Quick one-liner
    >>> colored = highlight_code("print('hello')", "python")

    >>> # Reusable highlighter
    >>> h = Highlighter(style="dracula")
    >>> for line in code.splitlines():
    ...     print(h.highlight_line(line, "rust"))
"""

from termflow.syntax.highlighter import (
    LANGUAGE_ALIASES,
    Highlighter,
    highlight_code,
    highlight_line,
)

__all__ = [
    "LANGUAGE_ALIASES",
    "Highlighter",
    "highlight_code",
    "highlight_line",
]
