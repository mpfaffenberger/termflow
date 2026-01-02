"""ANSI escape code constants and superscript utilities.

This module provides all the raw ANSI escape sequences needed for terminal
text styling, along with utilities for converting digits to Unicode superscripts.
"""

# =============================================================================
# Reset
# =============================================================================
RESET = "\x1b[0m"

# =============================================================================
# Text Styles - On/Off pairs
# =============================================================================
BOLD_ON = "\x1b[1m"
BOLD_OFF = "\x1b[22m"  # Also turns off DIM

DIM_ON = "\x1b[2m"
DIM_OFF = "\x1b[22m"  # Also turns off BOLD

ITALIC_ON = "\x1b[3m"
ITALIC_OFF = "\x1b[23m"

UNDERLINE_ON = "\x1b[4m"
UNDERLINE_OFF = "\x1b[24m"

STRIKEOUT_ON = "\x1b[9m"
STRIKEOUT_OFF = "\x1b[29m"

# =============================================================================
# Background
# =============================================================================
BGRESET = "\x1b[49m"

# =============================================================================
# Superscript Unicode Characters
# =============================================================================
SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def digit_to_superscript(digit: int) -> str:
    """Convert a digit (0-9) to its superscript Unicode character.

    Args:
        digit: An integer from 0 to 9.

    Returns:
        The corresponding superscript Unicode character.

    Raises:
        ValueError: If digit is not in range 0-9.

    Example:
        >>> digit_to_superscript(2)
        '²'
        >>> digit_to_superscript(0)
        '⁰'
    """
    if not 0 <= digit <= 9:
        raise ValueError(f"Digit must be 0-9, got {digit}")
    return SUPERSCRIPTS[digit]


def number_to_superscript(num: int) -> str:
    """Convert an integer to its superscript string representation.

    Handles negative numbers by prefixing with superscript minus (⁻).

    Args:
        num: Any integer.

    Returns:
        The superscript representation of the number.

    Example:
        >>> number_to_superscript(42)
        '⁴²'
        >>> number_to_superscript(-5)
        '⁻⁵'
        >>> number_to_superscript(0)
        '⁰'
    """
    if num < 0:
        return "⁻" + number_to_superscript(-num)
    if num == 0:
        return SUPERSCRIPTS[0]

    result = []
    while num > 0:
        result.append(SUPERSCRIPTS[num % 10])
        num //= 10
    return "".join(reversed(result))
