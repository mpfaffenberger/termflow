"""Text processing utilities for ANSI-escaped text.

This module provides critical utilities for handling text that contains
ANSI escape sequences, including:
- Stripping ANSI codes to get visible text
- Calculating visible width (handling CJK double-width characters)
- Splitting text into ANSI and non-ANSI segments
- ANSI-aware text wrapping that preserves styles across line breaks
"""

import re

from wcwidth import wcswidth, wcwidth

from termflow.ansi.codes import RESET

# =============================================================================
# Regex Patterns for ANSI Escape Sequences
# =============================================================================

#: Matches all ANSI escape sequences (CSI, OSC, etc.)
ANSI_ESCAPE_RE = re.compile(
    r"\x1b"
    r"(?:"
    r"\[[0-9;?]*[a-zA-Z]"  # CSI sequences: ESC [ ... letter
    r"|"
    r"\][0-9]*;[^\x1b]*(?:\x1b\\|\x07)"  # OSC sequences: ESC ] ... ST
    r"|"
    r"\[\?[0-9;]*[a-zA-Z]"  # Private CSI sequences
    r"|"
    r"[()][AB0-9]"  # Character set selection
    r")"
)

#: Matches SGR (Select Graphic Rendition) sequences specifically
ANSI_SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")

#: Matches any CSI sequence
ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def visible(text: str) -> str:
    """Remove all ANSI escape sequences, returning only visible text.

    Args:
        text: String potentially containing ANSI escape codes.

    Returns:
        String with all ANSI codes removed.

    Example:
        >>> visible("\x1b[1mBold\x1b[0m text")
        'Bold text'
        >>> visible("No codes here")
        'No codes here'
    """
    return ANSI_ESCAPE_RE.sub("", text)


def visible_length(text: str) -> int:
    """Calculate visible width of text (handles CJK double-width chars).

    Uses wcwidth library to correctly handle:
    - CJK characters (typically 2 cells wide)
    - Zero-width characters (combining marks, etc.)
    - Control characters

    Args:
        text: String potentially containing ANSI codes and wide characters.

    Returns:
        The display width in terminal cells.

    Example:
        >>> visible_length("\x1b[1mHello\x1b[0m")
        5
        >>> visible_length("你好")  # CJK characters are 2 cells each
        4
    """
    stripped = visible(text)
    width = wcswidth(stripped)
    # wcswidth returns -1 if string contains non-printable characters
    # In that case, fall back to counting characters individually
    if width < 0:
        width = sum(max(0, wcwidth(c)) for c in stripped)
    return width


def is_ansi_code(s: str) -> bool:
    """Check if string is an ANSI escape code.

    Args:
        s: String to check.

    Returns:
        True if the entire string is a single ANSI escape code.

    Example:
        >>> is_ansi_code("\x1b[1m")
        True
        >>> is_ansi_code("\x1b[1mtext")
        False
        >>> is_ansi_code("text")
        False
    """
    if not s.startswith("\x1b"):
        return False
    match = ANSI_ESCAPE_RE.match(s)
    return match is not None and match.group() == s


def split_ansi(text: str) -> list[str]:
    """Split text into alternating ANSI codes and text segments.

    The result alternates between regular text and ANSI escape codes.
    Empty strings are not included in the result.

    Args:
        text: String potentially containing ANSI escape codes.

    Returns:
        List of segments, alternating between text and ANSI codes.

    Example:
        >>> split_ansi("\x1b[1mBold\x1b[0m")
        ['\x1b[1m', 'Bold', '\x1b[0m']
        >>> split_ansi("plain text")
        ['plain text']
    """
    result = []
    last_end = 0

    for match in ANSI_ESCAPE_RE.finditer(text):
        # Add any text before this ANSI code
        if match.start() > last_end:
            result.append(text[last_end : match.start()])
        # Add the ANSI code
        result.append(match.group())
        last_end = match.end()

    # Add any remaining text after the last ANSI code
    if last_end < len(text):
        result.append(text[last_end:])

    return result


def extract_ansi_codes(text: str) -> list[str]:
    """Extract all ANSI escape codes from text.

    Args:
        text: String potentially containing ANSI escape codes.

    Returns:
        List of all ANSI escape codes found, in order.

    Example:
        >>> extract_ansi_codes("\x1b[1mBold\x1b[4mUnder\x1b[0m")
        ['\x1b[1m', '\x1b[4m', '\x1b[0m']
    """
    return ANSI_ESCAPE_RE.findall(text)


def parse_sgr_params(code: str) -> list[int]:
    """Parse SGR parameters from an ANSI code.

    SGR (Select Graphic Rendition) codes have the format ESC[n1;n2;...m
    This function extracts the numeric parameters.

    Args:
        code: An ANSI SGR escape code string.

    Returns:
        List of integer parameters, or [0] for reset if no params.

    Example:
        >>> parse_sgr_params("\x1b[1;4m")
        [1, 4]
        >>> parse_sgr_params("\x1b[38;2;255;128;0m")
        [38, 2, 255, 128, 0]
        >>> parse_sgr_params("\x1b[m")
        [0]
    """
    match = ANSI_SGR_RE.match(code)
    if not match:
        return []

    params_str = match.group(1)
    if not params_str:
        return [0]  # ESC[m is equivalent to ESC[0m

    return [int(p) for p in params_str.split(";") if p]


def _get_active_codes(segments: list[str]) -> str:
    """Get the cumulative ANSI codes that are currently active.

    Tracks SGR state through a sequence of segments, handling resets properly.

    Args:
        segments: List of text and ANSI code segments.

    Returns:
        Combined ANSI codes that should be active at the end.
    """
    active_codes: list[str] = []

    for segment in segments:
        if not is_ansi_code(segment):
            continue

        params = parse_sgr_params(segment)
        if not params:
            continue

        # Check for reset
        if 0 in params:
            active_codes.clear()
            # If there are other params after reset, keep processing
            if params == [0]:
                continue

        # Track non-reset codes
        active_codes.append(segment)

    return "".join(active_codes)


def wrap_ansi(text: str, width: int) -> list[str]:
    """Wrap text to width, preserving ANSI codes across line breaks.

    This is critical for proper text rendering! When a line is wrapped,
    any active ANSI styles are:
    1. Terminated at the end of each line with RESET
    2. Re-applied at the start of the next line

    Args:
        text: String potentially containing ANSI escape codes.
        width: Maximum visible width per line.

    Returns:
        List of wrapped lines, each with proper ANSI code handling.

    Example:
        >>> lines = wrap_ansi("\x1b[1mThis is bold text\x1b[0m", 10)
        >>> # Each line will have proper bold codes applied
    """
    if width <= 0:
        return [text] if text else []

    segments = split_ansi(text)
    lines: list[str] = []
    current_line: list[str] = []
    current_width = 0
    active_codes: list[str] = []  # Track currently active ANSI codes

    for segment in segments:
        if is_ansi_code(segment):
            # Track ANSI code state
            params = parse_sgr_params(segment)
            if params and 0 in params:
                # Reset clears active codes
                active_codes.clear()
            elif params:
                active_codes.append(segment)
            current_line.append(segment)
            continue

        # Process text segment character by character for proper width handling
        i = 0
        while i < len(segment):
            char = segment[i]
            char_width = max(0, wcwidth(char))

            # Check if this character would exceed line width
            if current_width + char_width > width and current_width > 0:
                # Finish current line
                if active_codes:
                    current_line.append(RESET)
                lines.append("".join(current_line))

                # Start new line with active codes
                current_line = list(active_codes)
                current_width = 0

            # Handle word wrapping at spaces
            if char == " " and current_width + char_width > width:
                i += 1
                continue  # Skip the space at line boundary

            current_line.append(char)
            current_width += char_width
            i += 1

    # Don't forget the last line
    if current_line:
        lines.append("".join(current_line))

    return lines if lines else [""]


def truncate_ansi(text: str, width: int, suffix: str = "…") -> str:
    """Truncate text to width, preserving ANSI codes and adding suffix.

    Args:
        text: String potentially containing ANSI escape codes.
        width: Maximum visible width.
        suffix: String to append when truncating (default: "…").

    Returns:
        Truncated string with ANSI codes properly handled.

    Example:
        >>> truncate_ansi("\x1b[1mHello World\x1b[0m", 8)
        '\x1b[1mHello W…\x1b[0m'
    """
    suffix_width = visible_length(suffix)
    if width <= suffix_width:
        return suffix[:width] if width > 0 else ""

    vis_len = visible_length(text)
    if vis_len <= width:
        return text

    target_width = width - suffix_width
    segments = split_ansi(text)
    result: list[str] = []
    current_width = 0
    active_codes: list[str] = []

    for segment in segments:
        if is_ansi_code(segment):
            params = parse_sgr_params(segment)
            if params and 0 in params:
                active_codes.clear()
            elif params:
                active_codes.append(segment)
            result.append(segment)
            continue

        for char in segment:
            char_width = max(0, wcwidth(char))
            if current_width + char_width > target_width:
                # Add suffix and reset
                result.append(suffix)
                if active_codes:
                    result.append(RESET)
                return "".join(result)
            result.append(char)
            current_width += char_width

    return "".join(result)
