"""Unified-diff parsing and theme-aware ANSI rendering.

Public API:

* :func:`parse_diff` / :func:`classify_line` / :func:`detect_language`
  -- structural parsing of unified diff text.
* :class:`DiffTheme` -- colors (backgrounds, marker brightness, tints).
* :class:`DiffRenderer` -- block rendering with syntax highlighting.
* :class:`DiffStream` -- incremental rendering for streamed diffs.
* :func:`render_diff` -- one-shot convenience.
* :func:`brighten_hex` -- small color helper used for marker accents.
"""

from termflow.diff.parser import (
    HEADER_PREFIXES,
    DiffKind,
    DiffLine,
    classify_line,
    detect_language,
    language_from_header,
    parse_diff,
)
from termflow.diff.renderer import (
    DiffRenderer,
    DiffStream,
    DiffTheme,
    brighten_hex,
    render_diff,
)

__all__ = [
    "HEADER_PREFIXES",
    "DiffKind",
    "DiffLine",
    "DiffRenderer",
    "DiffStream",
    "DiffTheme",
    "brighten_hex",
    "classify_line",
    "detect_language",
    "language_from_header",
    "parse_diff",
    "render_diff",
]
