"""Stock widgets for :class:`termflow.live.LiveApp` windows.

* :class:`PixelWidget` -- subclass and implement :meth:`~PixelWidget.render`
  to paint an RGB framebuffer every frame (games, visualizers).
* :class:`FramebufferView` -- shows frames pushed from elsewhere (an
  emulator, a native game engine, a video decoder), scaled to fit.
* :class:`TextLog` -- a file-like tail view; ``Renderer(output=log)``
  streams termflow markdown straight into a window.
* :class:`MarkdownView` -- a scrollable markdown document that can
  "type itself out" like a streaming LLM response.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING

from termflow.live.app import Widget
from termflow.live.pixels import PixelSurface
from termflow.render.document import render_markdown_lines
from termflow.tui.keys import Key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from termflow.live.buffer import Region
    from termflow.live.input import KeyEvent, KeyState
    from termflow.render.style import RenderStyle


class PixelWidget(Widget):
    """A widget backed by a :class:`PixelSurface` that tracks its size.

    The surface is ``region.width`` x ``2 * region.height`` pixels (two
    per cell). Override :meth:`render`; resolution changes with the
    window, so read ``surface.width`` / ``surface.height`` every frame.
    """

    def __init__(self) -> None:
        self.surface = PixelSurface(0, 0)

    def render(self, surface: PixelSurface) -> None:
        """Paint the current frame into ``surface``."""

    def draw(self, region: Region, focused: bool) -> None:  # noqa: ARG002
        self.surface.resize(region.width, region.height * 2)
        if self.surface.width and self.surface.height:
            self.render(self.surface)
            self.surface.draw(region)


class FramebufferView(PixelWidget):
    """Displays externally produced frames, nearest-neighbor scaled.

    Call :meth:`set_frame` from your engine (any thread) with a flat
    row-major sequence of ``0xRRGGBB`` pixels (high bits are ignored).

    Args:
        aspect: Display aspect ratio (width / height) to letterbox to,
            e.g. ``4 / 3`` for classic 320x200 games. None stretches the
            frame over the whole window.
        background: Color of the letterbox bars.
    """

    def __init__(self, aspect: float | None = None, background: int = 0) -> None:
        super().__init__()
        self.aspect = aspect
        self.background = background
        self._lock = threading.Lock()
        self._frame: tuple[Sequence[int], int, int] | None = None

    def set_frame(self, pixels: Sequence[int], width: int, height: int) -> None:
        with self._lock:
            self._frame = (pixels, width, height)

    def render(self, surface: PixelSurface) -> None:
        with self._lock:
            frame = self._frame
        if frame is None:
            return
        if self.aspect is None:
            surface.blit_scaled(*frame)
            return
        w, h = surface.width, surface.height
        fit_w, fit_h = min(w, round(h * self.aspect)), min(h, round(w / self.aspect))
        surface.clear(self.background)
        surface.blit_scaled(*frame, (w - fit_w) // 2, (h - fit_h) // 2, fit_w, fit_h)


class TextLog(Widget):
    """A thread-safe, file-like log that shows its newest lines.

    Lines may contain ANSI styling. ``write`` accepts partial lines, so
    a :class:`termflow.Renderer` can use it as its output stream.
    """

    focusable = False

    def __init__(self, max_lines: int = 1000) -> None:
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            *done, self._partial = (self._partial + text).split("\n")
            self._lines.extend(done)
        return len(text)

    def flush(self) -> None:
        """File-protocol no-op."""

    def lines(self) -> list[str]:
        with self._lock:
            return [*self._lines, self._partial] if self._partial else list(self._lines)

    def draw(self, region: Region, focused: bool) -> None:  # noqa: ARG002
        for y, line in enumerate(self.lines()[-region.height :] if region.height else []):
            region.ansi(0, y, line)


class MarkdownView(Widget):
    """A scrollable markdown document rendered by termflow.

    Args:
        markdown: The document.
        reveal_rate: Characters per second to "stream" the text in
            (None shows everything at once).
        style: Render style.

    While streaming, the view follows the newest text; Up/Down/PageUp/
    PageDown scroll when focused (scrolling up stops the follow).
    """

    def __init__(
        self,
        markdown: str = "",
        reveal_rate: float | None = None,
        style: RenderStyle | None = None,
    ) -> None:
        self._markdown = markdown
        self._rate = reveal_rate
        self._style = style
        self._revealed = float(len(markdown) if reveal_rate is None else 0)
        self._cache: tuple[int, int, list[str]] | None = None
        self._scroll = 0
        self._follow = True
        self._page = 1

    @property
    def done(self) -> bool:
        return int(self._revealed) >= len(self._markdown)

    def append(self, text: str) -> None:
        """Add more markdown (e.g. tokens from a model)."""
        self._markdown += text
        if self._rate is None:
            self._revealed = len(self._markdown)

    def update(self, dt: float, keys: KeyState) -> None:  # noqa: ARG002
        if self._rate is not None and not self.done:
            self._revealed = min(len(self._markdown), self._revealed + self._rate * dt)

    def on_key(self, event: KeyEvent) -> bool:
        step = {Key.UP: -1, Key.DOWN: 1, Key.PAGE_UP: -self._page, Key.PAGE_DOWN: self._page}
        if event.key not in step:
            return False
        self._scroll = max(0, self._scroll + step[event.key])
        self._follow = event.key in (Key.DOWN, Key.PAGE_DOWN) and self._follow
        return True

    def _rendered(self, width: int) -> list[str]:
        n = int(self._revealed)
        if self._cache is None or self._cache[:2] != (n, width):
            text = self._markdown[:n]
            lines = render_markdown_lines(text, width, style=self._style) if text else []
            self._cache = (n, width, lines)
        return self._cache[2]

    def draw(self, region: Region, focused: bool) -> None:  # noqa: ARG002
        if region.width <= 0 or region.height <= 0:
            return
        lines = self._rendered(region.width)
        self._page = max(1, region.height - 1)
        bottom = max(0, len(lines) - region.height)
        if self._follow or self._scroll >= bottom:
            self._scroll, self._follow = bottom, True
        for y, line in enumerate(lines[self._scroll : self._scroll + region.height]):
            region.ansi(0, y, line)
