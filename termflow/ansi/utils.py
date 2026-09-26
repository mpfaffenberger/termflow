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
    # int() pins the type: wcwidth ships without annotations -> Any.
    width = int(wcswidth(stripped))
    # wcswidth returns -1 if string contains non-printable characters
    # In that case, fall back to counting characters individually
    if width < 0:
        width = sum(max(0, int(wcwidth(c))) for c in stripped)
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


# =============================================================================
# SGR / Hyperlink State Tracking
# =============================================================================

#: SGR "off" attributes mapped to the "on" attributes they cancel.
_SGR_CANCELS: dict[int, frozenset[int]] = {
    22: frozenset({1, 2}),
    23: frozenset({3}),
    24: frozenset({4}),
    25: frozenset({5, 6}),
    27: frozenset({7}),
    28: frozenset({8}),
    29: frozenset({9}),
    39: frozenset({*range(30, 39), *range(90, 98)}),
    49: frozenset({*range(40, 49), *range(100, 108)}),
}

#: Closes an OSC 8 hyperlink (matches ``LINK[1]`` in :mod:`termflow.ansi.style`).
OSC8_CLOSE = "\x1b]8;;\x1b\\"

_OSC8_RE = re.compile(r"\x1b\]8;[^;]*;([^\x1b\x07]*)(?:\x1b\\|\x07)")


def _sgr_attributes(params: list[int]) -> list[int]:
    """Return the attribute selectors in ``params``, skipping color arguments.

    ``38;2;255;0;0`` is *one* attribute (38) -- its RGB arguments must not
    be mistaken for attributes like 0 (reset) or 22 (bold off).
    """
    attrs: list[int] = []
    i = 0
    while i < len(params):
        attr = params[i]
        attrs.append(attr)
        if attr in (38, 48, 58):
            mode = params[i + 1] if i + 1 < len(params) else None
            i += 5 if mode == 2 else 3 if mode == 5 else 2
        else:
            i += 1
    return attrs


def _track_sgr(active: list[str], code: str) -> None:
    """Update ``active`` (the SGR codes currently in effect) with ``code``.

    Resets clear everything, "off" codes (22, 24, 39, ...) drop the codes
    they cancel, and anything else is recorded. Non-SGR codes are ignored.
    """
    attrs = _sgr_attributes(parse_sgr_params(code))
    if not attrs:
        return
    if 0 in attrs:
        active.clear()
    cancelled = frozenset().union(*(_SGR_CANCELS.get(a, frozenset()) for a in attrs))
    if cancelled:
        active[:] = [c for c in active if not cancelled & set(_sgr_attributes(parse_sgr_params(c)))]
    if any(a != 0 and a not in _SGR_CANCELS for a in attrs):
        active.append(code)


def _osc8_opener(code: str) -> str | None:
    """Classify an OSC 8 hyperlink code.

    Returns the code itself if it opens a link, ``""`` if it closes one,
    and ``None`` if it is not an OSC 8 code at all.
    """
    match = _OSC8_RE.fullmatch(code)
    if match is None:
        return None
    return code if match.group(1) else ""


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
        if is_ansi_code(segment):
            _track_sgr(active_codes, segment)
    return "".join(active_codes)


# =============================================================================
# Wrapping / Truncation
# =============================================================================


def wrap_ansi(text: str, width: int, *, break_words: bool = False) -> list[str]:
    """Wrap text to width at word boundaries, preserving ANSI codes.

    Word boundaries (spaces/tabs) are preferred; a word only gets
    character-split if it is, by itself, longer than ``width``. Whitespace
    at a wrap point is dropped, so lines never end with a dangling space.

    When a line is wrapped, any active ANSI SGR styles (and any open OSC 8
    hyperlink) are:
    1. Terminated at the end of each line
    2. Re-applied at the start of the next line

    Args:
        text: String potentially containing ANSI escape codes.
        width: Maximum visible width per line.
        break_words: Split at exactly ``width`` cells instead of at word
            boundaries, keeping every space (use for code, where
            whitespace is significant).

    Returns:
        List of wrapped lines, each with proper ANSI code handling.

    Example:
        >>> lines = wrap_ansi("\x1b[1mThis is bold text\x1b[0m", 10)
        >>> # Each line will have proper bold codes applied
    """
    if width <= 0:
        return [text] if text else []

    lines: list[str] = []
    line: list[str] = []
    line_width = 0
    active: list[str] = []  # SGR codes in effect at the end of `line`
    link = ""  # OSC 8 opener in effect at the end of `line`

    word: list[str] = []  # buffered word: visible chars + embedded codes
    word_width = 0
    gap: list[str] = []  # whitespace between `line` and the buffered word

    def place(part: str) -> None:
        nonlocal line_width, link
        line.append(part)
        if is_ansi_code(part):
            opener = _osc8_opener(part)
            if opener is not None:
                link = opener
            else:
                _track_sgr(active, part)
        else:
            line_width += visible_length(part)

    def end_line() -> None:
        nonlocal line, line_width
        if link:
            line.append(OSC8_CLOSE)
        if active:
            line.append(RESET)
        lines.append("".join(line))
        line = [*active, link] if link else list(active)
        line_width = 0
        gap.clear()

    def emit_word() -> None:
        nonlocal word, word_width
        if not word:
            return
        gap_width = len(gap)
        if line_width + gap_width + word_width <= width:
            for part in (*gap, *word):
                place(part)
        elif word_width <= width and line_width > 0:
            end_line()
            for part in word:
                place(part)
        else:
            # Word is wider than the line -- fall back to character wrapping.
            if line_width > 0 and line_width + gap_width < width:
                for part in gap:
                    place(part)
            for part in word:
                if is_ansi_code(part):
                    place(part)
                    continue
                for ch in part:
                    if line_width + max(0, wcwidth(ch)) > width and line_width > 0:
                        end_line()
                    place(ch)
        gap.clear()
        word = []
        word_width = 0

    for segment in split_ansi(text):
        if is_ansi_code(segment):
            # Codes travel with the word so styling crosses a wrap boundary.
            word.append(segment)
            continue
        for ch in segment:
            if ch == "\n":
                emit_word()
                end_line()
            elif break_words:
                word.append(ch)
                word_width += max(0, wcwidth(ch))
                emit_word()
            elif ch in (" ", "\t"):
                emit_word()
                if line_width > 0:
                    gap.append(ch)
            else:
                word.append(ch)
                word_width += max(0, wcwidth(ch))

    emit_word()
    if line:
        lines.append("".join(line))

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
    result: list[str] = []
    current_width = 0
    active_codes: list[str] = []

    for segment in split_ansi(text):
        if is_ansi_code(segment):
            _track_sgr(active_codes, segment)
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
