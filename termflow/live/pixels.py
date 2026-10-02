"""RGB framebuffers rendered as half-block characters.

Each terminal cell shows two vertically stacked pixels: ``▀`` with the
top pixel as foreground and the bottom pixel as background. Truecolor
plus one boring glyph works in every modern terminal -- Windows
Terminal included -- with no graphics protocol negotiation.
"""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING

from termflow.live.buffer import UPPER_HALF

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from termflow.live.buffer import Region

#: Max pixel value without alpha/high bits.
_RGB_MASK = 0xFFFFFF


class PixelSurface:
    """A ``width`` x ``height`` grid of packed ``0xRRGGBB`` pixels.

    ``pixels`` is a flat row-major list; write to it directly in hot
    loops (``surface.pixels[y * surface.width + x] = color``).
    """

    def __init__(self, width: int, height: int, color: int = 0) -> None:
        self.width = max(0, width)
        self.height = max(0, height)
        self.pixels: list[int] = [color] * (self.width * self.height)
        # Scaling geometry -> C-speed gather of the sampled source pixels.
        self._gather: tuple[tuple[int, ...], Callable[[Sequence[int]], tuple[int, ...]]] | None = (
            None
        )

    def resize(self, width: int, height: int, color: int = 0) -> bool:
        """Reallocate if the size changed; returns True when it did."""
        width, height = max(0, width), max(0, height)
        if (width, height) == (self.width, self.height):
            return False
        self.width, self.height = width, height
        self.pixels = [color] * (width * height)
        return True

    def clear(self, color: int = 0) -> None:
        self.pixels[:] = [color] * len(self.pixels)

    def get(self, x: int, y: int) -> int:
        return self.pixels[y * self.width + x]

    def set(self, x: int, y: int, color: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y * self.width + x] = color

    def fill_rect(self, x: int, y: int, w: int, h: int, color: int) -> None:
        """Fill a rectangle (clipped)."""
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x0 >= x1:
            return
        run = [color] * (x1 - x0)
        for yy in range(y0, y1):
            start = yy * self.width
            self.pixels[start + x0 : start + x1] = run

    def blit_scaled(
        self,
        src: Sequence[int],
        src_width: int,
        src_height: int,
        x: int = 0,
        y: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Nearest-neighbor scale a foreign framebuffer into a rectangle.

        The integration point for fixed-resolution sources (an emulator,
        a 320x200 game engine, a video decoder). The destination defaults
        to the whole surface and is clipped to it. Only the low 24 bits of
        each source pixel are used, so ``0xAARRGGBB`` sources (e.g. a
        little-endian BGRA buffer read as uint32) work as-is.
        """
        w = self.width - x if width is None else width
        h = self.height - y if height is None else height
        if w <= 0 or h <= 0 or not (src_width and src_height):
            return
        x0, x1 = max(0, x), min(self.width, x + w)
        y0, y1 = max(0, y), min(self.height, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        geometry = (src_width, src_height, x, y, w, h, self.width, self.height)
        if self._gather is None or self._gather[0] != geometry:
            cols = [(dx - x) * src_width // w for dx in range(x0, x1)]
            indices = [
                ((dy - y) * src_height // h) * src_width + c for dy in range(y0, y1) for c in cols
            ]
            pick = itemgetter(*indices)
            gather = pick if len(indices) > 1 else (lambda s: (pick(s),))
            self._gather = (geometry, gather)
        values: Sequence[int] = self._gather[1](src)
        if max(values) > _RGB_MASK:  # ARGB source: drop the high byte
            values = [v & _RGB_MASK for v in values]
        run, stride = x1 - x0, self.width
        if run == stride:  # full-width destination: one slice assignment
            self.pixels[y0 * stride : y1 * stride] = values
            return
        for i, dy in enumerate(range(y0, y1)):
            start = dy * stride + x0
            self.pixels[start : start + run] = values[i * run : (i + 1) * run]

    def draw(self, region: Region) -> None:
        """Paint onto ``region`` -- one cell per two pixel rows."""
        w = self.width
        px = self.pixels
        glyphs = [UPPER_HALF] * w
        for cy in range((self.height + 1) // 2):
            top = 2 * cy * w
            bottom = top + w
            if bottom >= len(px):  # odd height: last row has no partner
                region.put_row(0, cy, glyphs, px[top : top + w], [0] * w)
            else:
                region.put_row(0, cy, glyphs, px[top:bottom], px[bottom : bottom + w])
