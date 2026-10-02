"""The live frame loop: windows, focus, layout, and a status bar.

A :class:`LiveApp` runs at a fixed frame rate. Every frame it polls
input, updates every widget, paints a fresh :class:`ScreenBuffer`, and
ships only the cells that changed::

    from termflow.live import LiveApp, Window, hsplit, MarkdownView, TextLog

    log = TextLog()
    app = LiveApp(
        [Window("Notes", MarkdownView("# hi")), Window("Log", log)],
        layout=lambda area: hsplit(area, 2, 1),
    )
    app.run()

Tab cycles focus; only the focused window sees held-key state. Ctrl+Q
or Ctrl+C quits (configurable via ``quit_keys``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from termflow.live.buffer import BOLD, Rect, Region, ScreenBuffer, render_diff, rgb
from termflow.live.console import LiveConsole
from termflow.live.input import KeyEvent, KeyState
from termflow.tui.keys import Key
from termflow.tui.terminal import terminal_size

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ACCENT = rgb(255, 140, 40)
MUTED = rgb(110, 110, 120)
STATUS_BG = rgb(30, 30, 38)
STATUS_FG = rgb(200, 200, 210)

#: Longest simulated step: a stall (window drag, breakpoint) should not
#: teleport the player through walls.
MAX_DT = 0.1


class Widget:
    """Base class for window contents; override what you need."""

    #: Whether Tab can focus this widget.
    focusable = True

    def on_key(self, event: KeyEvent) -> bool:  # noqa: ARG002 - override hook
        """Handle a discrete key event; return True to consume it."""
        return False

    def update(self, dt: float, keys: KeyState) -> None:
        """Advance ``dt`` seconds. ``keys`` is empty unless focused."""

    def draw(self, region: Region, focused: bool) -> None:
        """Paint into ``region`` (already clipped, inside the border)."""


@dataclass
class Window:
    """A titled, bordered pane hosting one widget."""

    title: str
    widget: Widget
    border: bool = True


@dataclass(frozen=True)
class AppStats:
    """What a finished :meth:`LiveApp.run` measured."""

    frames: int
    seconds: float
    bytes_written: int

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def _split(total: int, weights: Sequence[float]) -> list[tuple[int, int]]:
    """Divide ``total`` cells by weight -> (offset, size) pairs, no gaps."""
    s = sum(weights) or 1
    out, offset, acc = [], 0, 0.0
    for w in weights:
        acc += w
        end = round(total * acc / s)
        out.append((offset, end - offset))
        offset = end
    return out


def hsplit(area: Rect, *weights: float) -> list[Rect]:
    """Side-by-side columns sized by ``weights``."""
    return [Rect(area.x + o, area.y, n, area.height) for o, n in _split(area.width, weights)]


def vsplit(area: Rect, *weights: float) -> list[Rect]:
    """Stacked rows sized by ``weights``."""
    return [Rect(area.x, area.y + o, area.width, n) for o, n in _split(area.height, weights)]


class LiveApp:
    """A real-time, multi-window terminal app.

    Args:
        windows: The panes, in focus order.
        layout: Maps the usable area to one rect per window. Return
            fewer rects to hide trailing windows (e.g. on small terminals).
        title: Shown at the left of the status bar.
        fps: Target frame rate.
        quit_keys: Keys that end :meth:`run`.
        status: Optional callable for extra status-bar text.
        hints: Key hints shown after the title.
        console: Terminal session (a real :class:`LiveConsole` by default).
        size: Terminal size provider (columns, rows).
    """

    def __init__(
        self,
        windows: Sequence[Window],
        layout: Callable[[Rect], list[Rect]],
        *,
        title: str = "termflow live",
        fps: float = 30.0,
        quit_keys: Sequence[str] = ("ctrl-c", "ctrl-q"),
        status: Callable[[], str] | None = None,
        hints: str = "Tab focus · Ctrl+Q quit",
        console: LiveConsole | None = None,
        size: Callable[[], tuple[int, int]] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.windows = list(windows)
        self.layout = layout
        self.title = title
        self.fps = fps
        self.quit_keys = set(quit_keys)
        self.status = status
        self.hints = hints
        self.console = console or LiveConsole()
        self.keys = KeyState()
        self._idle_keys = KeyState()  # what unfocused widgets see: nothing
        self._size = size or terminal_size
        self._clock = clock
        self._sleep = sleep
        self._focus = next((i for i, w in enumerate(self.windows) if w.widget.focusable), 0)
        self._visible = len(self.windows)
        self.running = False
        self.measured_fps = 0.0

    @property
    def focused(self) -> Window | None:
        return self.windows[self._focus] if self.windows else None

    def focus(self, window: Window) -> None:
        self._focus = self.windows.index(window)
        self.keys.release_all()

    def focus_next(self) -> None:
        candidates = [i for i, w in enumerate(self.windows[: self._visible]) if w.widget.focusable]
        if not candidates:
            return
        later = [i for i in candidates if i > self._focus]
        self._focus = later[0] if later else candidates[0]
        self.keys.release_all()

    def quit(self) -> None:
        self.running = False

    def handle(self, events: Sequence[KeyEvent], now: float, reports_release: bool) -> None:
        """Route input: quit keys, then the focused widget, then Tab focus.

        The focused widget sees presses first and may consume them
        (a game that wants Tab for its own map). Unconsumed Tab cycles
        focus. Everything else also feeds the held-key state.
        """
        for event in events:
            if event.pressed and event.key in self.quit_keys:
                self.quit()
                continue
            focused = self.focused
            consumed = event.pressed and focused is not None and focused.widget.on_key(event)
            if event.pressed and not consumed and event.key == Key.TAB:
                self.focus_next()
                continue
            self.keys.feed(event, now, reports_release)

    def step(self, dt: float) -> ScreenBuffer:
        """Update every widget by ``dt`` and paint one frame."""
        dt = min(dt, MAX_DT)
        for i, window in enumerate(self.windows):
            window.widget.update(dt, self.keys if i == self._focus else self._idle_keys)
        cols, rows = self._size()
        buffer = ScreenBuffer(cols, rows)
        rects = self.layout(Rect(0, 0, cols, max(0, rows - 1)))
        self._visible = len(rects)
        if self._focus >= self._visible:
            self.focus_next()
        for i, (window, rect) in enumerate(zip(self.windows, rects, strict=False)):
            self._draw_window(buffer.region(rect), window, i == self._focus)
        self._draw_status(buffer.region(Rect(0, rows - 1, cols, 1)))
        return buffer

    def _draw_window(self, region: Region, window: Window, focused: bool) -> None:
        if not window.border:
            window.widget.draw(region, focused)
            return
        color = ACCENT if focused else MUTED
        region.box(window.title, fg=color, heavy=focused)
        inner = region.sub(Rect(1, 1, region.width - 2, region.height - 2))
        window.widget.draw(inner, focused)

    def _draw_status(self, region: Region) -> None:
        region.fill(bg=STATUS_BG)
        x = region.text(1, 0, self.title, ACCENT, STATUS_BG, BOLD)
        extra = self.status() if self.status else ""
        x = region.text(x, 0, f"  {self.hints}", STATUS_FG, STATUS_BG)
        if extra:
            region.text(x, 0, f"  │ {extra}", STATUS_FG, STATUS_BG)
        fps = f"{self.measured_fps:5.1f} fps "
        region.text(region.width - len(fps), 0, fps, MUTED, STATUS_BG)

    def run(self, max_frames: int | None = None) -> AppStats:
        """Run until a quit key (or ``max_frames``, for benchmarks)."""
        self.running = True
        frames = 0
        budget = 1.0 / self.fps
        with self.console as console:
            start = last = self._clock()
            prev: ScreenBuffer | None = None
            while self.running and (max_frames is None or frames < max_frames):
                now = self._clock()
                dt, last = now - last, now
                self.handle(console.poll(), now, console.reports_release)
                self.keys.expire(now)
                frame = self.step(dt)
                console.present(render_diff(prev, frame))
                prev = frame
                self.keys.end_frame()
                frames += 1
                if dt > 0:
                    self.measured_fps += (1.0 / dt - self.measured_fps) * 0.1
                spare = budget - (self._clock() - now)
                if spare > 0:
                    self._sleep(spare)
            elapsed = self._clock() - start
        return AppStats(frames, elapsed, self.console.bytes_written)
