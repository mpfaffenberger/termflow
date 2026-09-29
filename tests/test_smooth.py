"""Tests for termflow.stream (steady-rate smooth streaming)."""

import asyncio
from io import StringIO

import pytest

from termflow.stream import SmoothWriter, StreamSmoother, split_by_visible

# Fast timings so the suite doesn't dawdle.
FAST = {"tick_interval": 0.001, "catch_up_seconds": 0.005}


# =============================================================================
# split_by_visible
# =============================================================================
class TestSplitByVisible:
    def test_plain_text_split(self):
        emit, rest, consumed = split_by_visible("hello world", 5)
        assert emit == "hello"
        assert rest == " world"
        assert consumed == 5

    def test_budget_beyond_length(self):
        emit, rest, consumed = split_by_visible("hi", 10)
        assert emit == "hi"
        assert rest == ""
        assert consumed == 2

    def test_ansi_codes_are_free(self):
        s = "\x1b[1mab\x1b[0mcd"
        emit, rest, consumed = split_by_visible(s, 2)
        # The leading escape and the style-off escape ride along.
        assert emit == "\x1b[1mab\x1b[0m"
        assert rest == "cd"
        assert consumed == 2

    def test_trailing_escapes_attach_greedily(self):
        s = "x\x1b[0m\x1b[2m"
        emit, rest, consumed = split_by_visible(s, 1)
        assert emit == s
        assert rest == ""
        assert consumed == 1

    def test_zero_budget(self):
        emit, rest, consumed = split_by_visible("abc", 0)
        assert emit == ""
        assert rest == "abc"
        assert consumed == 0

    def test_empty_string(self):
        assert split_by_visible("", 5) == ("", "", 0)


# =============================================================================
# StreamSmoother
# =============================================================================
class TestStreamSmoother:
    def test_all_text_arrives_in_order(self):
        chunks: list[str] = []

        async def scenario():
            smoother = StreamSmoother(chunks.append, **FAST)
            smoother.start()
            smoother.feed("hello ")
            smoother.feed("world")
            await smoother.close()

        asyncio.run(scenario())
        assert "".join(chunks) == "hello world"

    def test_output_is_chunked_not_lumped(self):
        chunks: list[str] = []

        async def scenario():
            smoother = StreamSmoother(chunks.append, min_chars_per_tick=1, **FAST)
            smoother.start()
            smoother.feed("a" * 500)
            await smoother.close()

        asyncio.run(scenario())
        assert "".join(chunks) == "a" * 500
        assert len(chunks) > 1  # smoothed, not dumped in one write

    def test_burst_drains_linearly_within_catch_up_window(self):
        """A burst finishes in ~catch_up_ticks ticks, not a decaying tail.

        Re-deriving the quota from the shrinking remainder each tick makes
        the drain exponential: ``catch_up_ticks * ln(N / catch_up_ticks)``
        ticks of decay plus ``catch_up_ticks`` more at the 1-char floor.
        Each emitted chunk is one tick, so counting chunks pins the shape.
        """
        chunks: list[str] = []
        catch_up_ticks = 50

        async def scenario():
            smoother = StreamSmoother(
                chunks.append,
                tick_interval=0.001,
                catch_up_seconds=0.001 * catch_up_ticks,
                min_chars_per_tick=1,
            )
            smoother.start()
            smoother.feed("a" * 1000)
            await smoother.close()

        asyncio.run(scenario())
        assert "".join(chunks) == "a" * 1000
        assert len(chunks) <= catch_up_ticks + 1  # decay would need ~200

    def test_quota_resets_once_buffer_empties(self):
        """A trickle after a big burst is typed at its own pace, not the burst's."""
        chunks: list[str] = []

        async def scenario():
            smoother = StreamSmoother(
                chunks.append,
                tick_interval=0.001,
                catch_up_seconds=0.05,
                min_chars_per_tick=1,
            )
            smoother.start()
            smoother.feed("a" * 1000)
            while smoother._remaining_units():
                await asyncio.sleep(0.001)
            first_burst = len(chunks)
            smoother.feed("b" * 10)
            await smoother.close()
            return len(chunks) - first_burst

        trickle_chunks = asyncio.run(scenario())
        assert "".join(chunks) == "a" * 1000 + "b" * 10
        assert trickle_chunks >= 5  # stale quota of 20 would lump it in 1

    def test_abort_discards_backlog(self):
        chunks: list[str] = []

        async def scenario():
            smoother = StreamSmoother(chunks.append, **FAST)
            smoother.start()
            smoother.feed("x" * 10_000)
            smoother.abort()
            await asyncio.sleep(0.01)

        asyncio.run(scenario())
        assert "".join(chunks) != "x" * 10_000

    def test_pause_flushes_tail_atomically(self):
        chunks: list[str] = []
        paused = False

        async def scenario():
            nonlocal paused
            smoother = StreamSmoother(chunks.append, is_paused=lambda: paused, **FAST)
            smoother.start()
            smoother.feed("before-pause")
            await asyncio.sleep(0.003)
            paused = True
            await asyncio.sleep(0.01)
            # Everything fed so far must have landed by now.
            assert "".join(chunks) == "before-pause"
            paused = False
            await smoother.close()

        asyncio.run(scenario())

    def test_pause_hook_errors_are_swallowed(self):
        chunks: list[str] = []

        def broken() -> bool:
            raise RuntimeError("boom")

        async def scenario():
            smoother = StreamSmoother(chunks.append, is_paused=broken, **FAST)
            smoother.start()
            smoother.feed("still works")
            await smoother.close()

        asyncio.run(scenario())
        assert "".join(chunks) == "still works"

    def test_feed_during_pause_stays_silent_until_resume(self):
        chunks: list[str] = []
        paused = False

        async def scenario():
            nonlocal paused
            smoother = StreamSmoother(chunks.append, is_paused=lambda: paused, **FAST)
            smoother.start()
            smoother.feed("before ")
            await asyncio.sleep(0.01)
            paused = True
            await asyncio.sleep(0.005)
            smoother.feed("during")
            await asyncio.sleep(0.01)
            # Text fed while paused must NOT type over foreground output.
            assert "".join(chunks) == "before "
            paused = False
            await smoother.close()

        asyncio.run(scenario())
        assert "".join(chunks) == "before during"

    def test_close_without_start_is_safe(self):
        async def scenario():
            smoother = StreamSmoother(lambda _: None, **FAST)
            await smoother.close()

        asyncio.run(scenario())

    def test_emit_exceptions_do_not_hang_close(self):
        def explode(_chunk: str) -> None:
            raise RuntimeError("boom")

        async def scenario():
            smoother = StreamSmoother(explode, **FAST)
            smoother.start()
            smoother.feed("text")
            with pytest.raises(RuntimeError):
                await smoother.close()

        asyncio.run(scenario())


# =============================================================================
# SmoothWriter
# =============================================================================
class TestSmoothWriter:
    def test_file_like_interface(self):
        target = StringIO()
        writer = SmoothWriter(target, **FAST)
        assert writer.write("abc") == 3
        writer.flush()  # must not raise (and must not write eagerly)
        assert target.getvalue() == ""

    def test_drains_everything_on_close(self):
        target = StringIO()

        async def scenario():
            writer = SmoothWriter(target, **FAST)
            writer.start()
            writer.write("\x1b[1mBold\x1b[0m and plain")
            await writer.close()

        asyncio.run(scenario())
        assert target.getvalue() == "\x1b[1mBold\x1b[0m and plain"

    def test_ansi_never_split_across_ticks(self):
        target = StringIO()

        async def scenario():
            writer = SmoothWriter(target, min_chars_per_tick=1, **FAST)
            writer.start()
            writer.write("\x1b[38;2;1;2;3mcolored text here\x1b[0m")
            await writer.close()

        asyncio.run(scenario())
        # Every write must contain complete escape sequences: the final
        # buffer parses identically to the input.
        assert target.getvalue() == "\x1b[38;2;1;2;3mcolored text here\x1b[0m"

    def test_escape_split_across_writes(self):
        # An escape sequence arriving in two write() calls must still be
        # emitted atomically once complete.
        target = StringIO()

        async def scenario():
            writer = SmoothWriter(target, **FAST)
            writer.start()
            writer.write("\x1b[38;2;")
            writer.write("10;20;30mhi\x1b[0m")
            await writer.close()

        asyncio.run(scenario())
        assert target.getvalue() == "\x1b[38;2;10;20;30mhi\x1b[0m"

    def test_abort_stops_output(self):
        target = StringIO()

        async def scenario():
            writer = SmoothWriter(target, **FAST)
            writer.start()
            writer.write("y" * 10_000)
            writer.abort()
            await asyncio.sleep(0.01)

        asyncio.run(scenario())
        assert target.getvalue() != "y" * 10_000
