"""Text wrapping utilities with ANSI awareness.

Provides text wrapping that correctly handles:
- ANSI escape sequences (don't count towards width)
- CJK double-width characters
- Word boundaries
- Indentation and prefixes
"""

from __future__ import annotations

from termflow.ansi import visible_length, wrap_ansi


def text_wrap(
    text: str,
    width: int,
    indent: int = 0,
    first_prefix: str = "",
    cont_prefix: str = "",
) -> list[str]:
    """Wrap text to width with ANSI awareness.

    Args:
        text: Text to wrap (may contain ANSI codes)
        width: Maximum width per line
        indent: Additional indentation for wrapped lines
        first_prefix: Prefix for the first line
        cont_prefix: Prefix for continuation lines

    Returns:
        List of wrapped lines.

    Example:
        >>> lines = text_wrap("Hello world!", 10)
        >>> lines
        ['Hello', 'world!']
    """
    if not text:
        return [first_prefix] if first_prefix else []

    # Calculate available widths
    first_width = max(1, width - visible_length(first_prefix) - indent)
    cont_width = max(1, width - visible_length(cont_prefix) - indent)

    # Handle the case where we need different widths for first/cont
    if first_width == cont_width:
        # Simple case: uniform width
        lines = wrap_ansi(text, first_width)
    else:
        # Complex case: first line has different width
        lines = wrap_ansi(text, first_width)

        if len(lines) > 1:
            # Re-wrap remaining text with continuation width
            remaining = " ".join(lines[1:])
            cont_lines = wrap_ansi(remaining, cont_width)
            lines = [lines[0], *cont_lines]

    if not lines:
        return [first_prefix] if first_prefix else []

    # Apply prefixes
    result = []
    for i, line in enumerate(lines):
        if i == 0:
            result.append(f"{first_prefix}{line}")
        else:
            result.append(f"{cont_prefix}{' ' * indent}{line}")

    return result


def simple_wrap(text: str, width: int) -> list[str]:
    """Simple word wrap without ANSI handling.

    Faster than text_wrap when you know there are no ANSI codes.

    Args:
        text: Plain text to wrap
        width: Maximum width

    Returns:
        List of wrapped lines.
    """
    if not text or width <= 0:
        return [text] if text else []

    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        # +1 for space if not first word
        needed = word_len + (1 if current else 0)

        if current_len + needed <= width:
            current.append(word)
            current_len += needed
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
            current_len = word_len

    if current:
        lines.append(" ".join(current))

    return lines


def truncate(
    text: str,
    width: int,
    suffix: str = "…",
) -> str:
    """Truncate text to width, adding suffix if truncated.

    Args:
        text: Text to truncate
        width: Maximum width
        suffix: String to append when truncating

    Returns:
        Truncated text.
    """
    if visible_length(text) <= width:
        return text

    suffix_len = visible_length(suffix)
    target_width = width - suffix_len

    if target_width <= 0:
        return suffix[:width]

    # Use ANSI-aware truncation
    from termflow.ansi import truncate_ansi

    return truncate_ansi(text, width, suffix)


def pad_right(text: str, width: int, char: str = " ") -> str:
    """Pad text on the right to reach width.

    ANSI-aware: only counts visible characters.

    Args:
        text: Text to pad
        width: Target width
        char: Padding character

    Returns:
        Padded text.
    """
    current_width = visible_length(text)
    if current_width >= width:
        return text
    return text + char * (width - current_width)


def pad_left(text: str, width: int, char: str = " ") -> str:
    """Pad text on the left to reach width.

    ANSI-aware: only counts visible characters.

    Args:
        text: Text to pad
        width: Target width
        char: Padding character

    Returns:
        Padded text.
    """
    current_width = visible_length(text)
    if current_width >= width:
        return text
    return char * (width - current_width) + text


def center(text: str, width: int, char: str = " ") -> str:
    """Center text within width.

    ANSI-aware: only counts visible characters.

    Args:
        text: Text to center
        width: Target width
        char: Padding character

    Returns:
        Centered text.
    """
    current_width = visible_length(text)
    if current_width >= width:
        return text

    total_padding = width - current_width
    left_pad = total_padding // 2
    right_pad = total_padding - left_pad

    return char * left_pad + text + char * right_pad
