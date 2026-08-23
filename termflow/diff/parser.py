"""Unified-diff parsing primitives.

Turns raw unified-diff text into structured :class:`DiffLine` records and
sniffs the source language from ``---``/``+++`` headers so renderers can
apply proper syntax highlighting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DiffKind = Literal["add", "remove", "context", "header"]

#: Prefixes that mark unified-diff metadata rather than file content.
HEADER_PREFIXES: tuple[str, ...] = ("---", "+++", "@@", "diff ", "index ")

#: Matches ``--- a/path/file.ext`` / ``+++ b/path/file.ext`` headers and
#: captures the file extension (used for language detection).
_HEADER_EXTENSION_RE = re.compile(r"^(?:\+\+\+|---) [ab]/.*?\.([A-Za-z0-9]+)$")


@dataclass(frozen=True)
class DiffLine:
    """One classified line of a unified diff.

    Attributes:
        kind: ``"add"``, ``"remove"``, ``"context"``, or ``"header"``.
        content: Line content with the leading diff marker stripped.
        raw: The original, unmodified line.
    """

    kind: DiffKind
    content: str
    raw: str


def classify_line(line: str) -> DiffLine:
    """Classify a single unified-diff line.

    Headers keep their raw text as content; add/remove lines lose their
    ``+``/``-`` marker; context lines lose their leading space.
    """
    if line.startswith(HEADER_PREFIXES):
        return DiffLine(kind="header", content=line, raw=line)
    if line.startswith("+"):
        return DiffLine(kind="add", content=line[1:], raw=line)
    if line.startswith("-"):
        return DiffLine(kind="remove", content=line[1:], raw=line)
    content = line[1:] if line.startswith(" ") else line
    return DiffLine(kind="context", content=content, raw=line)


def parse_diff(diff_text: str) -> list[DiffLine]:
    """Parse unified diff text into classified lines.

    A single trailing empty line (from a trailing newline) is dropped,
    matching how diffs are conventionally displayed.
    """
    if not diff_text:
        return []
    lines = diff_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [classify_line(line) for line in lines]


def language_from_header(line: str) -> str | None:
    """Extract a language hint (file extension) from one diff header line.

    Returns the bare extension (e.g. ``"py"``) or None when the line is
    not a file header / has no extension. The extension flows through
    :data:`termflow.syntax.highlighter.LANGUAGE_ALIASES`, so common
    extensions map to their canonical lexer automatically.
    """
    match = _HEADER_EXTENSION_RE.match(line)
    return match.group(1).lower() if match else None


def detect_language(diff_text: str, max_lines: int = 10) -> str:
    """Sniff the source language from a diff's file headers.

    Args:
        diff_text: Unified diff text.
        max_lines: How many leading lines to inspect.

    Returns:
        A language identifier suitable for :class:`~termflow.syntax.Highlighter`
        (falls back to ``"text"``).
    """
    for line in diff_text.split("\n")[:max_lines]:
        language = language_from_header(line)
        if language:
            return language
    return "text"
