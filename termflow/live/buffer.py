"""Cell grid, clipped drawing regions, and a diffing frame renderer.

The live mode never prints lines; it paints a :class:`ScreenBuffer`
every frame and ships only what changed since the previous frame
(:func:`render_diff`). That is what makes 30 fps over a terminal pipe
feasible -- a game view repaints most cells, but borders, side panes,
and status bars cost nothing once drawn.

Colors are packed ``0xRRGGBB`` ints (truecolor); :data:`DEFAULT` means
"the terminal's own default color".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from operator import ne

from wcwidth import wcwidth

#: wcwidth is pure but not cheap; UIs reuse a small alphabet every frame.
_char_width = lru_cache(maxsize=4096)(wcwidth)

#: "Use the terminal's default color" sentinel for fg/bg.
DEFAULT = -1

# Attribute bit flags.
BOLD = 1
DIM = 2
ITALIC = 4
UNDERLINE = 8
REVERSE = 16

_ATTR_SGR = ((BOLD, "1"), (DIM, "2"), (ITALIC, "3"), (UNDERLINE, "4"), (REVERSE, "7"))

#: Placeholder stored in the cell to the right of a double-width char.
WIDE_TAIL = ""

#: Half-block pixel glyphs: two vertically stacked pixels per cell.
UPPER_HALF = "\u2580"  # ▀ fg = top pixel, bg = bottom pixel
LOWER_HALF = "\u2584"  # ▄ fg = bottom pixel, bg = top pixel

_BOX_LIGHT = ("┌", "┐", "└", "┘", "─", "│")
_BOX_HEAVY = ("┏", "┓", "┗", "┛", "━", "┃")

# SGR (we interpret), OSC (hyperlinks etc., skipped), other CSI (skipped).
_ANSI_RE = re.compile(
    r"\x1b\[([0-9;:]*)m|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;?]*[ -/]*[@-~]"
)

_BASIC_COLORS = (
    0x000000, 0xCD3131, 0x0DBC79, 0xE5E510, 0x2472C8, 0xBC3FBC, 0x11A8CD, 0xE5E5E5,
    0x666666, 0xF14C4C, 0x23D18B, 0xF5F543, 0x3B8EEA, 0xD670D6, 0x29B8DB, 0xFFFFFF,
)  # fmt: skip


def rgb(r: int, g: int, b: int) -> int:
    """Pack 0-255 channels into a ``0xRRGGBB`` color int."""
    return (r & 255) << 16 | (g & 255) << 8 | (b & 255)


def _xterm256(n: int) -> int:
    if n < 16:
        return _BASIC_COLORS[n]
    if n < 232:
        n -= 16
        steps = (0, 95, 135, 175, 215, 255)
        return rgb(steps[n // 36], steps[n // 6 % 6], steps[n % 6])
    level = 8 + (n - 232) * 10
    return rgb(level, level, level)


@dataclass(frozen=True)
class Rect:
    """A screen rectangle in cell coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def inset(self, n: int = 1) -> Rect:
        """Shrink by ``n`` cells on every side (never below zero size)."""
        return Rect(self.x + n, self.y + n, max(0, self.width - 2 * n), max(0, self.height - 2 * n))


class ScreenBuffer:
    """A ``width`` x ``height`` grid of styled cells (flat, row-major).

    Storage is four parallel flat lists -- the cheapest shape for the
    slice comparisons :func:`render_diff` leans on. Draw through
    :meth:`region` rather than poking the lists directly.
    """

    def __init__(self, width: int, height: int, bg: int = DEFAULT) -> None:
        self.width = max(0, width)
        self.height = max(0, height)
        n = self.width * self.height
        self.chars: list[str] = [" "] * n
        self.fg: list[int] = [DEFAULT] * n
        self.bg: list[int] = [bg] * n
        self.attrs: list[int] = [0] * n

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def region(self, rect: Rect | None = None) -> Region:
        """A clipped drawing surface over ``rect`` (default: everything)."""
        rect = rect or Rect(0, 0, self.width, self.height)
        return Region(self, rect)

    def row_text(self, y: int) -> str:
        """Plain text of one row (handy in tests)."""
        start = y * self.width
        return "".join(self.chars[start : start + self.width])


class Region:
    """A clipped, offset view of a :class:`ScreenBuffer`.

    Coordinates are local to the region; anything outside is silently
    clipped, so widgets never need to bounds-check their own drawing.
    """

    def __init__(self, buffer: ScreenBuffer, rect: Rect) -> None:
        # Clip the rect itself to the buffer so every write is safe.
        x0, y0 = max(0, rect.x), max(0, rect.y)
        x1 = min(buffer.width, rect.x + rect.width)
        y1 = min(buffer.height, rect.y + rect.height)
        self.buffer = buffer
        self.rect = Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))
        # Local origin stays where the caller asked, even if clipped.
        self._ox, self._oy = rect.x, rect.y

    @property
    def width(self) -> int:
        return self.rect.x + self.rect.width - self._ox

    @property
    def height(self) -> int:
        return self.rect.y + self.rect.height - self._oy

    def sub(self, rect: Rect) -> Region:
        """A nested region; ``rect`` is local to this one and clipped to it."""
        abs_rect = Rect(self._ox + rect.x, self._oy + rect.y, rect.width, rect.height)
        child = Region(self.buffer, abs_rect)
        # Clip to the parent too, not just the buffer.
        x0, y0 = max(child.rect.x, self.rect.x), max(child.rect.y, self.rect.y)
        x1 = min(child.rect.x + child.rect.width, self.rect.x + self.rect.width)
        y1 = min(child.rect.y + child.rect.height, self.rect.y + self.rect.height)
        child.rect = Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))
        return child

    def _index(self, x: int, y: int) -> int | None:
        ax, ay = self._ox + x, self._oy + y
        r = self.rect
        if r.x <= ax < r.x + r.width and r.y <= ay < r.y + r.height:
            return ay * self.buffer.width + ax
        return None

    def set(
        self, x: int, y: int, char: str, fg: int = DEFAULT, bg: int = DEFAULT, attrs: int = 0
    ) -> None:
        """Set one cell (clipped)."""
        i = self._index(x, y)
        if i is not None:
            b = self.buffer
            # Writing a wide char's tail right after its lead must not
            # "repair" that brand-new lead away.
            _unorphan(b, i, i + 1, keep_lead=char == WIDE_TAIL)
            b.chars[i], b.fg[i], b.bg[i], b.attrs[i] = char, fg, bg, attrs

    def fill(self, char: str = " ", fg: int = DEFAULT, bg: int = DEFAULT, attrs: int = 0) -> None:
        """Fill the whole region."""
        b, r = self.buffer, self.rect
        for ay in range(r.y, r.y + r.height):
            start = ay * b.width + r.x
            end = start + r.width
            _unorphan(b, start, end)
            b.chars[start:end] = [char] * r.width
            b.fg[start:end] = [fg] * r.width
            b.bg[start:end] = [bg] * r.width
            b.attrs[start:end] = [attrs] * r.width

    def _clip_run(self, x: int, y: int, n: int) -> tuple[int, int, int, int] | None:
        """Clip an ``n``-cell run at (x, y): (buffer start, end, src lo, src hi).

        Also blanks any wide char the run would cut in half.
        """
        ay = self._oy + y
        r = self.rect
        if not r.y <= ay < r.y + r.height:
            return None
        ax = self._ox + x
        lo, hi = max(ax, r.x), min(ax + n, r.x + r.width)
        if lo >= hi:
            return None
        start = ay * self.buffer.width + lo
        end = start + (hi - lo)
        _unorphan(self.buffer, start, end)
        return start, end, lo - ax, hi - ax

    def put_row(self, x: int, y: int, chars: list[str], fg: list[int], bg: list[int]) -> None:
        """Bulk-write a run of cells (the fast path for pixel blits)."""
        run = self._clip_run(x, y, len(chars))
        if run is None:
            return
        start, end, s0, s1 = run
        b = self.buffer
        b.chars[start:end] = chars[s0:s1]
        b.fg[start:end] = fg[s0:s1]
        b.bg[start:end] = bg[s0:s1]
        b.attrs[start:end] = [0] * (end - start)

    def text(
        self, x: int, y: int, text: str, fg: int = DEFAULT, bg: int = DEFAULT, attrs: int = 0
    ) -> int:
        """Write plain ``text`` at (x, y); returns the x after the last cell."""
        if (text.isascii() and text.isprintable()) or all(_char_width(c) == 1 for c in text):
            # Fast path: one cell per char, so the whole run is a slice write.
            run = self._clip_run(x, y, len(text))
            if run is not None:
                start, end, s0, s1 = run
                n, b = end - start, self.buffer
                b.chars[start:end] = text[s0:s1]
                b.fg[start:end] = [fg] * n
                b.bg[start:end] = [bg] * n
                b.attrs[start:end] = [attrs] * n
            return x + len(text)
        for ch in text:
            x = self._put_char(x, y, ch, fg, bg, attrs)
        return x

    def ansi(self, x: int, y: int, text: str, bg: int = DEFAULT) -> int:
        """Write ANSI-styled ``text`` (e.g. termflow-rendered markdown).

        Interprets SGR colors (16/256/truecolor) and bold/dim/italic/
        underline/reverse; skips OSC sequences (hyperlinks) and other
        CSI controls. Returns the x after the last cell.
        """
        fg, cur_bg, attrs = DEFAULT, bg, 0
        pos = 0
        for m in _ANSI_RE.finditer(text):
            if m.start() > pos:
                x = self.text(x, y, text[pos : m.start()], fg, cur_bg, attrs)
            pos = m.end()
            if m.group(1) is not None:
                fg, cur_bg, attrs = _apply_sgr(m.group(1), fg, cur_bg, attrs, bg)
        if pos < len(text):
            x = self.text(x, y, text[pos:], fg, cur_bg, attrs)
        return x

    def _put_char(self, x: int, y: int, ch: str, fg: int, bg: int, attrs: int) -> int:
        w: int = _char_width(ch)
        if w < 0:  # control char: ignore
            return x
        if w == 0:  # combining mark: attach to the previous cell
            i = self._index(x - 1, y)
            if i is not None:
                self.buffer.chars[i] += ch
            return x
        if w == 2 and self._index(x + 1, y) is None:
            # Half-visible wide char would corrupt the neighbor: blank it.
            self.set(x, y, " ", fg, bg, attrs)
            return x + 2
        self.set(x, y, ch, fg, bg, attrs)
        if w == 2:
            self.set(x + 1, y, WIDE_TAIL, fg, bg, attrs)
        return x + w

    def box(
        self, title: str = "", fg: int = DEFAULT, heavy: bool = False, title_fg: int | None = None
    ) -> None:
        """Draw a border around the region with an optional title."""
        w, h = self.width, self.height
        if w < 2 or h < 2:
            return
        tl, tr, bl, br, hz, vt = _BOX_HEAVY if heavy else _BOX_LIGHT
        self.text(0, 0, tl + hz * (w - 2) + tr, fg)
        self.text(0, h - 1, bl + hz * (w - 2) + br, fg)
        for yy in range(1, h - 1):
            self.set(0, yy, vt, fg)
            self.set(w - 1, yy, vt, fg)
        if title and w > 6:
            label = f" {title} "[: w - 4]
            self.text(2, 0, label, fg if title_fg is None else title_fg, attrs=BOLD)


def _unorphan(b: ScreenBuffer, start: int, end: int, keep_lead: bool = False) -> None:
    """Before overwriting cells [start, end) of one row, blank any wide
    char that would lose one of its halves -- terminals do the same, and
    a stale half would throw :func:`render_diff`'s cursor math off."""
    row_start = start - start % b.width
    if not keep_lead and start > row_start and b.chars[start] == WIDE_TAIL:
        b.chars[start - 1] = " "
    if end < row_start + b.width and b.chars[end] == WIDE_TAIL:
        b.chars[end] = " "


def _apply_sgr(params: str, fg: int, bg: int, attrs: int, base_bg: int) -> tuple[int, int, int]:
    codes = [int(p) if p.isdigit() else 0 for p in params.replace(":", ";").split(";")] or [0]
    i = 0
    while i < len(codes):
        c = codes[i]
        if c == 0:
            fg, bg, attrs = DEFAULT, base_bg, 0
        elif c in (1, 2, 3, 4, 7):
            attrs |= {1: BOLD, 2: DIM, 3: ITALIC, 4: UNDERLINE, 7: REVERSE}[c]
        elif c == 22:
            attrs &= ~(BOLD | DIM)
        elif c == 23:
            attrs &= ~ITALIC
        elif c == 24:
            attrs &= ~UNDERLINE
        elif c == 27:
            attrs &= ~REVERSE
        elif 30 <= c <= 37 or 90 <= c <= 97:
            fg = _BASIC_COLORS[c - 30 if c < 90 else c - 82]
        elif 40 <= c <= 47 or 100 <= c <= 107:
            bg = _BASIC_COLORS[c - 40 if c < 100 else c - 92]
        elif c == 39:
            fg = DEFAULT
        elif c == 49:
            bg = base_bg
        elif c in (38, 48) and i + 1 < len(codes):
            color, used = _extended_color(codes, i + 1)
            if color is not None:
                fg, bg = (color, bg) if c == 38 else (fg, color)
            i += used
        i += 1
    return fg, bg, attrs


def _extended_color(codes: list[int], i: int) -> tuple[int | None, int]:
    """Parse ``5;n`` or ``2;r;g;b``; returns (color, params consumed)."""
    if codes[i] == 5 and i + 1 < len(codes):
        return _xterm256(codes[i + 1]), 2
    if codes[i] == 2 and i + 3 < len(codes):
        return rgb(codes[i + 1], codes[i + 2], codes[i + 3]), 4
    return None, 1


def _color_param(base: int, color: int) -> str:
    """``38;2;r;g;b`` / ``48;2;r;g;b`` (``39`` / ``49`` for DEFAULT)."""
    if color == DEFAULT:
        return str(base + 1)
    return f"{base};2;{color >> 16 & 255};{color >> 8 & 255};{color & 255}"


class _ParamCache(dict[int, str]):
    """color -> SGR parameter text, memoized (bounded: smooth gradients
    could otherwise grow it forever)."""

    def __init__(self, base: int) -> None:
        super().__init__()
        self.base = base

    def __missing__(self, color: int) -> str:
        if len(self) > 65536:
            self.clear()
        value = self[color] = _color_param(self.base, color)
        return value


_FG = _ParamCache(38)
_BG = _ParamCache(48)


class _SeqCache(dict[int, str]):
    """color -> complete ``ESC[...m`` sequence for one component."""

    def __init__(self, params: _ParamCache) -> None:
        super().__init__()
        self.params = params

    def __missing__(self, color: int) -> str:
        if len(self) > 65536:
            self.clear()
        value = self[color] = f"\x1b[{self.params[color]}m"
        return value


_FG_SEQ = _SeqCache(_FG)
_BG_SEQ = _SeqCache(_BG)


def _encode_pixels(
    tops: list[int], bottoms: list[int], lf: int, lb: int, out: list[str]
) -> tuple[int, int]:
    """Encode a run of half-block pixel cells; returns the final (fg, bg).

    The pure-pixel fast path of :func:`render_diff`: no attributes, no
    default colors, no wide chars -- so each cell is just (top, bottom)
    and a decision table picks ▀, ▄, or a space, whichever needs the
    fewest color changes given the terminal's current (fg, bg).
    """
    append = out.append
    fgs, bgs, fg_params, bg_params = _FG_SEQ, _BG_SEQ, _FG, _BG
    up, down = UPPER_HALF, LOWER_HALF
    for t, b in zip(tops, bottoms, strict=True):
        if t == b:  # solid: a space shows only the background
            if b != lb:
                append(bgs[b])
                lb = b
            append(" ")
        elif t == lf:
            if b != lb:
                append(bgs[b])
                lb = b
            append(up)
        elif b == lf:  # ▄ draws the bottom pixel with the current fg
            if t != lb:
                append(bgs[t])
                lb = t
            append(down)
        elif b == lb:
            append(fgs[t])
            lf = t
            append(up)
        elif t == lb:
            append(fgs[b])
            lf = b
            append(down)
        else:
            append(f"\x1b[{fg_params[t]};{bg_params[b]}m{up}")
            lf, lb = t, b
    return lf, lb


def _full_sgr(fg: int, bg: int, attrs: int) -> str:
    """Reset, then set attributes and both colors."""
    parts = ["0"] + [code for flag, code in _ATTR_SGR if attrs & flag]
    if fg != DEFAULT:
        parts.append(_FG[fg])
    if bg != DEFAULT:
        parts.append(_BG[bg])
    return f"\x1b[{';'.join(parts)}m"


def _dirty_span(prev: ScreenBuffer, cur: ScreenBuffer, a: int, b: int) -> tuple[int, int] | None:
    """First and one-past-last differing index within [a, b) (None: clean).

    Element-wise comparison and the searches all run in C
    (``map(ne, ...)``, ``in``, ``.index``): no Python call per cell.
    """
    lo, hi = b, a
    for old, new in (
        (prev.chars, cur.chars),
        (prev.fg, cur.fg),
        (prev.bg, cur.bg),
        (prev.attrs, cur.attrs),
    ):
        old_row, new_row = old[a:b], new[a:b]
        if old_row == new_row:
            continue
        mask = list(map(ne, old_row, new_row))
        lo = min(lo, a + mask.index(True))
        hi = max(hi, b - mask[::-1].index(True))
    return (lo, hi) if lo < hi else None


def render_diff(prev: ScreenBuffer | None, cur: ScreenBuffer) -> str:
    """ANSI that turns a terminal showing ``prev`` into one showing ``cur``.

    ``prev=None`` (or a size change) repaints everything. Otherwise each
    changed row rewrites only the span between its first and last dirty
    cell -- one cursor move per row, SGR only when the style changes.
    Assumes autowrap is off (see :class:`termflow.live.console.LiveConsole`).
    """
    full = prev is None or prev.size != cur.size
    out: list[str] = ["\x1b[0m\x1b[2J"] if full else []
    width = cur.width
    # The terminal's current style. Every frame ends with a reset and full
    # repaints start with one, so each frame begins in the default style.
    lf, lb, la = DEFAULT, DEFAULT, 0
    chars, fgs, bgs, attrs = cur.chars, cur.fg, cur.bg, cur.attrs
    fg_param, bg_param = _FG, _BG
    for y in range(cur.height):
        a, b = y * width, (y + 1) * width
        if full:
            lo, hi = a, b
        else:
            span = _dirty_span(prev, cur, a, b)  # type: ignore[arg-type]
            if span is None:
                continue
            lo, hi = span
            # Never start a span on the right half of a wide char.
            while lo > a and chars[lo] == WIDE_TAIL:
                lo -= 1
        out.append(f"\x1b[{y + 1};{lo - a + 1}H")
        tops, bottoms = fgs[lo:hi], bgs[lo:hi]
        if (
            chars[lo:hi].count(UPPER_HALF) == hi - lo
            and not any(attrs[lo:hi])
            and min(tops) >= 0
            and min(bottoms) >= 0
        ):  # a pure pixel span (a game view row): specialized encoder
            if la != 0:
                out.append("\x1b[0m")
                lf, lb, la = DEFAULT, DEFAULT, 0
            lf, lb = _encode_pixels(tops, bottoms, lf, lb, out)
            continue
        # Hot loop (every pixel cell of a game view goes through here):
        # local names, zip instead of indexing, memoized SGR parameters,
        # and only the color components that actually change.
        for ch, fg, bg, at in zip(chars[lo:hi], fgs[lo:hi], bgs[lo:hi], attrs[lo:hi], strict=True):
            if ch == WIDE_TAIL:
                continue  # the terminal already advanced past it
            if at == 0 and fg >= 0 and bg >= 0:  # plain cell, explicit colors
                if ch == UPPER_HALF:
                    if fg == bg:
                        ch = " "  # solid: only the background matters
                    elif (bg != lf) + (fg != lb) < (fg != lf) + (bg != lb):
                        # Same two pixels as ▄ with colors swapped: fewer changes.
                        ch, fg, bg = LOWER_HALF, bg, fg
                if ch == " " and la == 0:  # a space never shows its fg
                    if bg != lb:
                        out.append(f"\x1b[{bg_param[bg]}m")
                        lb = bg
                    out.append(" ")
                    continue
            if at != la:
                out.append(_full_sgr(fg, bg, at))
                lf, lb, la = fg, bg, at
            elif fg != lf:
                seq = f"{fg_param[fg]};{bg_param[bg]}" if bg != lb else fg_param[fg]
                out.append(f"\x1b[{seq}m")
                lf, lb = fg, bg
            elif bg != lb:
                out.append(f"\x1b[{bg_param[bg]}m")
                lb = bg
            out.append(ch)
    if out:
        out.append("\x1b[0m")
    return "".join(out)
