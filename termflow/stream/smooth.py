"""Adaptive-rate drain engine for smooth streamed terminal output.

Bursty producers (LLM token streams especially) make jerky terminals.
The classes here buffer incoming content and release it via a background
asyncio task at a steady, adaptive rate: each tick releases a slice
proportional to the backlog, aiming to fully drain over a short catch-up
window. Latency stays low when the producer races ahead, while output
still feels smooth when it trickles.
"""

from __future__ import annotations

import asyncio
import math
from typing import IO, TYPE_CHECKING

from termflow.ansi.utils import ANSI_ESCAPE_RE

if TYPE_CHECKING:
    from collections.abc import Callable


class SteadyDrainer:
    """Drive a background task that drains a buffer at an adaptive rate.

    Subclasses implement the buffer mechanics via :meth:`_remaining_units`,
    :meth:`_drain_units`, and :meth:`_flush_all`. A "unit" is whatever the
    subclass counts as one step of progress (e.g. a visible character).

    Args:
        tick_interval: Seconds between drain ticks.
        catch_up_seconds: Window over which the current backlog should be
            fully drained (bigger = smoother, laggier).
        min_units_per_tick: Floor on units released per tick.
        is_paused: Optional callable polled each tick; while it returns
            True the drainer flushes its tail once (so pending output
            lands atomically) and then stays silent until unpaused.
    """

    def __init__(
        self,
        *,
        tick_interval: float = 0.02,
        catch_up_seconds: float = 0.4,
        min_units_per_tick: int = 1,
        is_paused: Callable[[], bool] | None = None,
    ) -> None:
        self._tick = tick_interval
        # Ticks over which we aim to drain the current backlog.
        self._catch_up_ticks = max(1, round(catch_up_seconds / tick_interval))
        self._min_units = max(1, min_units_per_tick)
        self._is_paused = is_paused or (lambda: False)
        self._closed = False
        self._task: asyncio.Task | None = None
        self._pending = ""

    def start(self) -> None:
        """Spin up the background drain task (idempotent)."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        """Mark the stream finished and wait for the buffer to fully drain."""
        self._closed = True
        task, self._task = self._task, None
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            # We were cancelled while waiting (user interrupt). Make sure
            # the drain task dies with us and nothing prints afterwards.
            task.cancel()
            self._discard_all()
            raise

    def abort(self) -> None:
        """Stop immediately and discard buffered content (user interrupt).

        Unlike :meth:`close`, nothing further is printed: the user asked
        us to stop, so dumping the backlog would just be noise.
        """
        self._closed = True
        self._discard_all()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()

    async def _run(self) -> None:
        try:
            was_paused = self._paused()
            while True:
                if self._paused():
                    if not was_paused:
                        # Pause just began: flush the tail in ONE atomic
                        # write so it lands before whatever the pause is
                        # for, and close() returns immediately instead of
                        # stalling the producer.
                        self._flush_all()
                    was_paused = True
                    if self._closed and self._remaining_units() <= 0:
                        return
                    # Anything fed DURING the pause stays silent until
                    # resume so we never type over foreground output.
                    await asyncio.sleep(self._tick)
                    continue
                was_paused = False
                remaining = self._remaining_units()
                if remaining <= 0:
                    if self._closed:
                        return
                    await asyncio.sleep(self._tick)
                    continue
                n = max(
                    self._min_units,
                    math.ceil(remaining / self._catch_up_ticks),
                )
                self._drain_units(n)
                await asyncio.sleep(self._tick)
        except asyncio.CancelledError:
            # Cancellation means interrupt/shutdown: stop typing NOW and
            # drop the backlog instead of dumping it into the terminal.
            self._discard_all()
            raise

    def _discard_all(self) -> None:
        """Throw away any buffered content without emitting it."""
        self._pending = ""

    def _paused(self) -> bool:
        """Best-effort poll of the injected pause hook."""
        try:
            return bool(self._is_paused())
        except Exception:
            return False

    # -- subclass hooks -----------------------------------------------------
    def _remaining_units(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def _drain_units(self, n: int) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _flush_all(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class StreamSmoother(SteadyDrainer):
    """Buffer plain-text deltas and emit them at a consistent rate.

    Emission is delegated to a callback so callers can wrap chunks in
    whatever styling machinery they like (dim ANSI codes, a Rich console,
    a logging sink...) without this class knowing about any of it.

    Example:
        >>> smoother = StreamSmoother(lambda chunk: print(chunk, end=""))
        >>> smoother.start()
        >>> smoother.feed("streamed text")
    """

    def __init__(
        self,
        emit: Callable[[str], None],
        *,
        tick_interval: float = 0.02,
        catch_up_seconds: float = 0.4,
        min_chars_per_tick: int = 2,
        is_paused: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            tick_interval=tick_interval,
            catch_up_seconds=catch_up_seconds,
            min_units_per_tick=min_chars_per_tick,
            is_paused=is_paused,
        )
        self._emit = emit

    def feed(self, text: str) -> None:
        """Append streamed text to the buffer."""
        if text:
            self._pending += text

    def _remaining_units(self) -> int:
        return len(self._pending)

    def _drain_units(self, n: int) -> None:
        chunk, self._pending = self._pending[:n], self._pending[n:]
        self._emit(chunk)

    def _flush_all(self) -> None:
        if self._pending:
            self._emit(self._pending)
            self._pending = ""


class SmoothWriter(SteadyDrainer):
    """File-like proxy that types pre-rendered ANSI text out smoothly.

    A :class:`~termflow.render.renderer.Renderer` (or anything else)
    writes ANSI-styled text to this object; the background drainer then
    releases it to ``target`` one visible character at a time. ANSI
    escape sequences are emitted atomically (and greedily attached to
    the preceding character) so styling never breaks mid-code.
    """

    def __init__(
        self,
        target: IO[str],
        *,
        tick_interval: float = 0.012,
        catch_up_seconds: float = 0.5,
        min_chars_per_tick: int = 1,
        is_paused: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            tick_interval=tick_interval,
            catch_up_seconds=catch_up_seconds,
            min_units_per_tick=min_chars_per_tick,
            is_paused=is_paused,
        )
        self._target = target

    # -- file-like interface used by Renderer -------------------------------
    def write(self, text: str) -> int:
        if text:
            self._pending += text
        return len(text)

    def flush(self) -> None:
        # Real flushing is owned by the drain task; renderers flush eagerly
        # after every write, but we want to control the cadence ourselves.
        pass

    # -- drainer hooks ------------------------------------------------------
    def _remaining_units(self) -> int:
        # Count visible chars from the live buffer so escape sequences split
        # across write() boundaries can't desync a cached counter.
        return len(ANSI_ESCAPE_RE.sub("", self._pending))

    def _drain_units(self, n: int) -> None:
        emit, rest, _ = split_by_visible(self._pending, n)
        if not emit:
            return
        self._pending = rest
        self._target.write(emit)
        self._target.flush()

    def _flush_all(self) -> None:
        if self._pending:
            self._target.write(self._pending)
            self._target.flush()
            self._pending = ""


def split_by_visible(s: str, budget: int) -> tuple[str, str, int]:
    """Split ``s`` after ``budget`` visible chars, keeping ANSI codes atomic.

    Returns ``(emit, rest, consumed_visible)`` where ``emit`` contains
    exactly ``consumed_visible`` visible characters plus any ANSI escape
    sequences that surround them (trailing escapes are greedily attached
    so style-off codes flush together with their text).
    """
    i = 0
    consumed = 0
    n = len(s)
    while i < n and consumed < budget:
        m = ANSI_ESCAPE_RE.match(s, i)
        if m:
            i = m.end()
        else:
            i += 1
            consumed += 1
    # Greedily attach any trailing escape sequences.
    while True:
        m = ANSI_ESCAPE_RE.match(s, i)
        if not m:
            break
        i = m.end()
    return s[:i], s[i:], consumed
