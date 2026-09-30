"""Swappable, live agent transcripts (not a terminal emulator).

Each producer writes styled, newline-delimited text to its own buffer. The
viewer alone owns terminal output; hidden agents keep collecting history.
Cursor movement and other terminal-control sequences are not supported content.
"""

from __future__ import annotations

from io import SEEK_END, StringIO, UnsupportedOperation
from threading import RLock
from typing import IO, TYPE_CHECKING

from termflow.tui.keys import Key
from termflow.tui.pager import Pager, PagerResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from termflow.render.style import RenderStyle


class HistoryBuffer(StringIO):
    """Thread-safe, append-only text sink usable as a Renderer output.

    History is retained in memory until the buffer is discarded. Snapshots
    remain available after closing; closing prevents further writes.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = RLock()
        self._closed_value = ""

    def writable(self) -> bool:
        return not self.closed

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("history expects text")
        with self._lock:
            if self.closed:
                raise ValueError("write to closed history")
            super().seek(0, SEEK_END)
            return super().write(text)

    def seekable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = 0) -> int:  # noqa: ARG002
        raise UnsupportedOperation("history is append-only")

    def truncate(self, size: int | None = None) -> int:  # noqa: ARG002
        raise UnsupportedOperation("history is append-only")

    def close(self) -> None:
        with self._lock:
            if not self.closed:
                self._closed_value = super().getvalue()
                super().close()

    def lines(self) -> list[str]:
        """Return an independent snapshot, including any unfinished line."""
        return self.getvalue().split("\n")

    def getvalue(self) -> str:
        """Return the complete transcript without changing it."""
        with self._lock:
            return self._closed_value if self.closed else super().getvalue()


class AgentHistory:
    """Ordered collection of independent agent output buffers.

    Register agents before handing their buffers to producers. Registration
    and snapshots are safe from worker threads; viewer methods belong to the
    thread running the viewer.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._buffers: dict[str, HistoryBuffer] = {}

    def add(self, agent_id: str) -> HistoryBuffer:
        """Register a unique, nonempty ID and return its output sink."""
        if not agent_id:
            raise ValueError("agent ID must not be empty")
        with self._lock:
            if agent_id in self._buffers:
                raise ValueError(f"agent already registered: {agent_id}")
            buffer = HistoryBuffer()
            self._buffers[agent_id] = buffer
            return buffer

    @property
    def agent_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._buffers)

    def buffer(self, agent_id: str) -> HistoryBuffer:
        """Look up an output sink; unknown IDs raise KeyError."""
        with self._lock:
            return self._buffers[agent_id]


class AgentHistoryViewer(Pager):
    """Live read-only viewer with independent scroll/follow state per agent.

    Tab / Right selects the next agent, Left the previous one. Pager scrolling
    and close keys work normally. End / G resumes following new output; scrolling
    away from the bottom stops following. Poll ticks refresh even without input.
    Use ``use_alt_screen=False`` inside an existing ``terminal_session``.
    Custom key handlers can inspect ``active_agent`` to route host actions.
    """

    def __init__(
        self,
        history: AgentHistory,
        *,
        agent_id: str | None = None,
        style: RenderStyle | None = None,
        output: IO[str] | None = None,
        key_source: Callable[[], str] | None = None,
        size: Callable[[], tuple[int, int]] | None = None,
        use_alt_screen: bool = True,
        key_handlers: dict[str, Callable[[Pager], PagerResult | None]] | None = None,
    ) -> None:
        ids = history.agent_ids
        if not ids:
            raise ValueError("register at least one agent before opening history")
        self._history = history
        self._active_agent = agent_id if agent_id is not None else ids[0]
        history.buffer(self._active_agent)
        self._positions: dict[str, tuple[int, bool]] = {}
        self._following = True
        super().__init__(
            self._active_agent,
            style=style,
            output=output,
            key_source=key_source,
            size=size,
            use_alt_screen=use_alt_screen,
            key_handlers=key_handlers,
            footer_hint="Tab/Left/Right agent - j/k scroll - End follow - q close",
        )
        self._sync()

    @property
    def active_agent(self) -> str:
        return self._active_agent

    def select(self, agent_id: str) -> None:
        """Switch histories, preserving the previous agent's viewport."""
        self._history.buffer(agent_id)  # Validate before changing state.
        if agent_id == self._active_agent:
            return
        self._positions[self._active_agent] = (self._top, self._following)
        self._active_agent = agent_id
        self._top, self._following = self._positions.get(agent_id, (0, True))
        self._sync()

    def _sync(self) -> None:
        self._title = self._active_agent
        self._lines = self._history.buffer(self._active_agent).lines()
        self._top = self._max_top() if self._following else min(self._top, self._max_top())

    def _frame(self) -> list[str]:
        self._sync()
        return super()._frame()

    def scroll(self, delta: int) -> None:
        super().scroll(delta)
        self._following = self._top == self._max_top()

    def _wait_key(self) -> str:
        # Unlike a static pager, repaint on every polling tick for live output.
        return self._read_key()

    def _handle_key(self, key: str) -> PagerResult | None:
        if key not in self._key_handlers:
            if key in (Key.TAB, Key.RIGHT, Key.LEFT):
                ids = self._history.agent_ids
                step = -1 if key == Key.LEFT else 1
                self.select(ids[(ids.index(self._active_agent) + step) % len(ids)])
                return None
            if key in (Key.END, "G", Key.HOME, "g"):
                self._following = key in (Key.END, "G")
        return super()._handle_key(key)
