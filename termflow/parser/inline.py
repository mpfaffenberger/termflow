"""Inline markdown parser.

Handles parsing of inline markdown formatting:
- **bold**, *italic*, ***bold italic***
- __underline__ (when not in words)
- ~~strikethrough~~
- `inline code` and ``code with backticks``
- [links](url) and ![images](url)
- [^footnotes]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from termflow.parser.entities import decode_if_needed


class InlineElement(Enum):
    """Inline element types for internal tokenization."""

    TEXT = auto()
    BOLD = auto()
    ITALIC = auto()
    BOLD_ITALIC = auto()
    UNDERLINE = auto()
    STRIKEOUT = auto()
    CODE = auto()
    LINK = auto()
    IMAGE = auto()
    FOOTNOTE = auto()


@dataclass(slots=True)
class InlineToken:
    """A parsed inline element."""

    element_type: InlineElement
    content: str
    url: str | None = None  # For links/images

    def __repr__(self) -> str:
        if self.url:
            return f"InlineToken({self.element_type.name}, {self.content!r}, {self.url!r})"
        return f"InlineToken({self.element_type.name}, {self.content!r})"


# =============================================================================
# Regex Patterns
# =============================================================================

# Code spans - match backticks with content
# Handles both `code` and ``code with `backticks` ``
CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")

# Links: [text](url) - non-greedy matching
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# Images: ![alt](url)
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Footnotes: [^N] or [^word]
FOOTNOTE_RE = re.compile(r"\[\^([^\]]+)\]")

# Bold italic: ***text*** or ___text___
BOLD_ITALIC_RE = re.compile(r"\*{3}(.+?)\*{3}|_{3}(.+?)_{3}")

# Bold: **text** (but not ***)
BOLD_RE = re.compile(r"(?<!\*)\*{2}(?!\*)(.+?)(?<!\*)\*{2}(?!\*)")

# Italic with asterisk: *text* (but not ** or ***)
ITALIC_ASTERISK_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")

# Italic with underscore: _text_ (not in word context)
# Must have word boundary or start/end
ITALIC_UNDERSCORE_RE = re.compile(r"(?<![\w])_([^_]+)_(?![\w])")

# Underline: __text__ (double underscore, not in word context)
UNDERLINE_RE = re.compile(r"(?<![\w])__([^_]+)__(?![\w])")

# Strikethrough: ~~text~~
STRIKEOUT_RE = re.compile(r"~~(.+?)~~")


class InlineParser:
    """Parses inline markdown formatting.

    Handles: **bold**, *italic*, ***bold italic***, __underline__,
    ~~strikeout~~, `code`, [links](url), ![images](url), [^footnotes]

    Example:
        >>> parser = InlineParser()
        >>> tokens = parser.parse("Hello **world**!")
        >>> [t.element_type.name for t in tokens]
        ['TEXT', 'BOLD', 'TEXT']
    """

    def __init__(self, process_links: bool = True, process_images: bool = True):
        """Initialize the inline parser.

        Args:
            process_links: Whether to parse [links](url).
            process_images: Whether to parse ![images](url).
        """
        self.process_links = process_links
        self.process_images = process_images

    def parse(self, text: str) -> list[InlineToken]:
        """Parse a line and return inline tokens.

        Args:
            text: The text to parse.

        Returns:
            List of InlineToken objects.
        """
        if not text:
            return []

        # Decode HTML entities first
        text = decode_if_needed(text)

        # Use a span-based approach: find all formatting spans, sort by position,
        # then extract tokens in order
        spans: list[tuple[int, int, InlineElement, str, str | None]] = []

        # Find all code spans first (they take precedence and escape other formatting)
        code_positions: set[int] = set()
        for match in CODE_SPAN_RE.finditer(text):
            start, end = match.start(), match.end()
            content = match.group(2)
            spans.append((start, end, InlineElement.CODE, content, None))
            code_positions.update(range(start, end))

        def is_in_code(pos: int) -> bool:
            return pos in code_positions

        # Find images (before links to handle ![...](...))
        if self.process_images:
            for match in IMAGE_RE.finditer(text):
                if not is_in_code(match.start()):
                    spans.append(
                        (
                            match.start(),
                            match.end(),
                            InlineElement.IMAGE,
                            match.group(1),
                            match.group(2),
                        )
                    )

        # Find links
        if self.process_links:
            for match in LINK_RE.finditer(text):
                if not is_in_code(match.start()):
                    # Make sure this isn't an image (starts with !)
                    if match.start() > 0 and text[match.start() - 1] == "!":
                        continue
                    spans.append(
                        (
                            match.start(),
                            match.end(),
                            InlineElement.LINK,
                            match.group(1),
                            match.group(2),
                        )
                    )

        # Find footnotes
        for match in FOOTNOTE_RE.finditer(text):
            if not is_in_code(match.start()):
                # Make sure this isn't a link [text](url)
                if match.end() < len(text) and text[match.end()] == "(":
                    continue
                spans.append(
                    (match.start(), match.end(), InlineElement.FOOTNOTE, match.group(1), None)
                )

        # Find bold italic (***)
        for match in BOLD_ITALIC_RE.finditer(text):
            if not is_in_code(match.start()):
                content = match.group(1) or match.group(2)
                spans.append((match.start(), match.end(), InlineElement.BOLD_ITALIC, content, None))

        # Find strikethrough
        for match in STRIKEOUT_RE.finditer(text):
            if not is_in_code(match.start()):
                spans.append(
                    (match.start(), match.end(), InlineElement.STRIKEOUT, match.group(1), None)
                )

        # Find underline (__ before bold check)
        for match in UNDERLINE_RE.finditer(text):
            if not is_in_code(match.start()):
                spans.append(
                    (match.start(), match.end(), InlineElement.UNDERLINE, match.group(1), None)
                )

        # Find bold (**)
        for match in BOLD_RE.finditer(text):
            if not is_in_code(match.start()):
                spans.append((match.start(), match.end(), InlineElement.BOLD, match.group(1), None))

        # Find italic (*)
        for match in ITALIC_ASTERISK_RE.finditer(text):
            if not is_in_code(match.start()):
                spans.append(
                    (match.start(), match.end(), InlineElement.ITALIC, match.group(1), None)
                )

        # Find italic (_) - only at word boundaries
        for match in ITALIC_UNDERSCORE_RE.finditer(text):
            if not is_in_code(match.start()):
                spans.append(
                    (match.start(), match.end(), InlineElement.ITALIC, match.group(1), None)
                )

        # Remove overlapping spans (keep the first one found)
        spans = self._remove_overlapping_spans(spans)

        # Sort by start position
        spans.sort(key=lambda x: x[0])

        # Build token list, filling in TEXT tokens for gaps
        tokens: list[InlineToken] = []
        pos = 0

        for start, end, elem_type, content, url in spans:
            # Add text before this span
            if start > pos:
                tokens.append(InlineToken(InlineElement.TEXT, text[pos:start]))

            # Add the formatted token
            tokens.append(InlineToken(elem_type, content, url))
            pos = end

        # Add remaining text
        if pos < len(text):
            tokens.append(InlineToken(InlineElement.TEXT, text[pos:]))

        return tokens

    def _remove_overlapping_spans(
        self, spans: list[tuple[int, int, InlineElement, str, str | None]]
    ) -> list[tuple[int, int, InlineElement, str, str | None]]:
        """Remove overlapping spans, keeping earlier/longer ones."""
        if not spans:
            return spans

        # Sort by start position, then by length (longer first)
        spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        result: list[tuple[int, int, InlineElement, str, str | None]] = []
        occupied: set[int] = set()

        for span in spans:
            start, end = span[0], span[1]
            # Check if any position is already occupied
            if not any(p in occupied for p in range(start, end)):
                result.append(span)
                occupied.update(range(start, end))

        return result

    def parse_simple(self, text: str) -> str:
        """Parse and return just the visible text content.

        Strips all formatting and returns plain text.

        Args:
            text: The text to parse.

        Returns:
            Plain text with formatting removed.
        """
        tokens = self.parse(text)
        return "".join(t.content for t in tokens)


def tokenize_inline(text: str) -> list[InlineToken]:
    """Convenience function to tokenize inline markdown.

    Args:
        text: Text to tokenize.

    Returns:
        List of inline tokens.
    """
    return InlineParser().parse(text)


def strip_formatting(text: str) -> str:
    """Strip all inline formatting, returning plain text.

    Args:
        text: Markdown text.

    Returns:
        Plain text without formatting.
    """
    return InlineParser().parse_simple(text)
