"""Theme-aware ANSI rendering of unified diffs, block or streaming.

The renderer pairs :mod:`termflow.diff.parser` with the syntax
highlighter: added/removed lines get colored backgrounds and bold
markers, code tokens keep their highlighted foregrounds, and themes can
nudge per-line-type foreground tints.

Block usage:
    >>> from termflow.diff import DiffRenderer
    >>> print(DiffRenderer().render(diff_text))

Streaming usage (chunks arrive over time):
    >>> from termflow.diff import DiffStream
    >>> stream = DiffStream()
    >>> for chunk in chunks:
    ...     sys.stdout.write(stream.feed(chunk))
    >>> sys.stdout.write(stream.close())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from termflow.ansi.codes import BOLD_ON, RESET
from termflow.ansi.color import bg_color, fg_color, hex2rgb, rgb2hex
from termflow.ansi.utils import ANSI_SGR_RE
from termflow.diff.parser import (
    DiffLine,
    classify_line,
    language_from_header,
)
from termflow.syntax import Highlighter

#: Truecolor foreground SGR — rewritten when a theme applies line tints.
_TRUECOLOR_FG_RE = re.compile(r"\x1b\[38;2;(\d+);(\d+);(\d+)m")


def brighten_hex(hex_color: str, factor: float) -> str:
    """Scale each RGB channel of ``#RRGGBB`` by ``(1 + factor)``.

    factor=0.0 -> no change; positive brightens, negative darkens.
    Channels clamp to [0, 255].

    Raises:
        ValueError: If ``hex_color`` is not a valid ``#RRGGBB`` string.
    """
    rgb = hex2rgb(hex_color)
    if rgb is None:
        raise ValueError(f"Expected #RRGGBB, got {hex_color!r}")
    r, g, b = (max(0, min(255, int(channel * (1 + factor)))) for channel in rgb)
    return rgb2hex(r, g, b).lower()


@dataclass
class DiffTheme:
    """Colors driving diff rendering.

    Attributes:
        addition: Background hex for added lines.
        deletion: Background hex for removed lines.
        marker_brighten: How much brighter than the background the
            ``+``/``-`` markers render.
        line_tints: Optional per-kind RGB deltas (e.g.
            ``{"added": (0, 24, 0)}``) shifting highlighted foreground
            colors so token colors harmonize with the line background.
            Keys: ``"added"``, ``"removed"``, ``"context"``.
    """

    addition: str = "#0e4429"
    deletion: str = "#67060c"
    marker_brighten: float = 0.6
    line_tints: dict[str, tuple[int, int, int]] = field(default_factory=dict)

    @property
    def addition_marker(self) -> str:
        """Foreground hex for the ``+`` marker."""
        return brighten_hex(self.addition, self.marker_brighten)

    @property
    def deletion_marker(self) -> str:
        """Foreground hex for the ``-`` marker."""
        return brighten_hex(self.deletion, self.marker_brighten)


#: Maps DiffLine.kind to the tint-dictionary keys themes use.
_KIND_TO_TINT_KEY = {"add": "added", "remove": "removed", "context": "context"}


class DiffRenderer:
    """Render unified diffs to ANSI with syntax highlighting.

    Args:
        highlighter: Syntax highlighter to use. May expose a
            ``diff_line_tints`` attribute (same shape as
            :attr:`DiffTheme.line_tints`) which is honored when the
            theme itself sets no tints — this lets themed highlighters
            carry their own diff accents.
        theme: Diff colors; defaults to :class:`DiffTheme`.
        show_headers: Render ``---``/``+++``/``@@`` headers (dimmed)
            instead of skipping them.
    """

    def __init__(
        self,
        highlighter: Highlighter | None = None,
        theme: DiffTheme | None = None,
        show_headers: bool = False,
    ) -> None:
        self.highlighter = highlighter or Highlighter()
        self.theme = theme or DiffTheme()
        self.show_headers = show_headers
        if not self.theme.line_tints:
            tints = getattr(self.highlighter, "diff_line_tints", None)
            if isinstance(tints, dict):
                self.theme.line_tints = tints

    # -- public API --------------------------------------------------------

    def render(self, diff_text: str, language: str | None = None) -> str:
        """Render a complete diff to an ANSI string (no trailing newline)."""
        stream = DiffStream(self, language=language)
        rendered = stream.feed(diff_text) + stream.close()
        return rendered.rstrip("\n")

    def render_line(self, line: DiffLine, language: str = "text") -> str | None:
        """Render one classified diff line to ANSI.

        Returns None for skipped lines (headers when ``show_headers``
        is off), and ``""`` for blank lines.
        """
        if line.kind == "header":
            if not self.show_headers:
                return None
            return f"\x1b[2m{line.raw}{RESET}"
        if not line.raw:
            return ""
        if line.kind == "add":
            return self._render_marked_line(
                line, "+ ", self.theme.addition, self.theme.addition_marker, language
            )
        if line.kind == "remove":
            return self._render_marked_line(
                line, "- ", self.theme.deletion, self.theme.deletion_marker, language
            )
        return f"  {self._highlight(line, language)}"

    # -- internals ---------------------------------------------------------

    def _render_marked_line(
        self,
        line: DiffLine,
        marker: str,
        bg_hex: str,
        marker_fg_hex: str,
        language: str,
    ) -> str:
        bg = bg_color(bg_hex)
        prefix = f"{BOLD_ON}{fg_color(marker_fg_hex)}{bg}{marker}{RESET}"
        return f"{prefix}{self._with_background(self._highlight(line, language), bg)}"

    def _highlight(self, line: DiffLine, language: str) -> str:
        highlighted = self.highlighter.highlight_line(line.content, language)
        tint = self.theme.line_tints.get(_KIND_TO_TINT_KEY[line.kind])
        return _apply_tint(highlighted, tint) if tint else highlighted

    @staticmethod
    def _with_background(ansi_text: str, bg: str) -> str:
        """Lay ``ansi_text`` over a background, surviving embedded resets."""
        if not bg:
            return ansi_text
        reasserted = ANSI_SGR_RE.sub(lambda m: m.group(0) + bg, ansi_text)
        return f"{bg}{reasserted}{RESET}"


def _apply_tint(ansi_text: str, tint: tuple[int, int, int]) -> str:
    """Shift every truecolor foreground in ``ansi_text`` by ``tint``.

    Only 24-bit foregrounds are rewritten; 256-color output passes
    through untouched (tints are a truecolor nicety, not a contract).
    """

    def shift(match: re.Match[str]) -> str:
        channels = (
            max(0, min(255, int(value) + delta))
            for value, delta in zip(match.groups(), tint, strict=True)
        )
        r, g, b = channels
        return f"\x1b[38;2;{r};{g};{b}m"

    return _TRUECOLOR_FG_RE.sub(shift, ansi_text)


class DiffStream:
    """Incrementally render diff text as it arrives.

    Feed arbitrary chunks; complete lines render immediately (each with
    a trailing newline) while partial lines buffer until finished.
    Language is sniffed live from file headers, so multi-file diffs
    re-highlight when the file (and extension) changes mid-stream.

    Args:
        renderer: Renderer to use (a default one is created if omitted).
        language: Explicit language override; disables header sniffing.
    """

    def __init__(
        self,
        renderer: DiffRenderer | None = None,
        language: str | None = None,
    ) -> None:
        self._renderer = renderer or DiffRenderer()
        self._language = language
        self._sniff = language is None
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """Consume a chunk; return ANSI for any completed lines."""
        self._buffer += chunk
        rendered: list[str] = []
        while "\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition("\n")
            piece = self._render_one(line)
            if piece is not None:
                rendered.append(piece + "\n")
        return "".join(rendered)

    def close(self) -> str:
        """Flush the trailing partial line (no trailing newline)."""
        if not self._buffer:
            return ""
        piece, self._buffer = self._render_one(self._buffer), ""
        return piece or ""

    def _render_one(self, raw_line: str) -> str | None:
        line = classify_line(raw_line)
        if self._sniff and line.kind == "header":
            self._language = language_from_header(line.raw) or self._language
        return self._renderer.render_line(line, language=self._language or "text")


def render_diff(
    diff_text: str,
    language: str | None = None,
    theme: DiffTheme | None = None,
    highlighter: Highlighter | None = None,
) -> str:
    """One-shot convenience: render a unified diff to ANSI."""
    return DiffRenderer(highlighter=highlighter, theme=theme).render(diff_text, language=language)
