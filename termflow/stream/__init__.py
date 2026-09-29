"""Steady-rate streaming helpers for buttery-smooth terminal output.

Models (and other bursty producers) emit text in lumps: a big chunk, a
pause, another chunk. Printing each delta the instant it lands makes
output stutter and jerk. This package buffers incoming content and
releases it at an adaptive, consistent rate — like double-buffering in
a video game, but for your terminal.

* :class:`SteadyDrainer` -- the abstract pacing engine.
* :class:`StreamSmoother` -- plain-text smoother that emits via callback.
* :class:`SmoothWriter` -- file-like proxy that types pre-rendered ANSI
  text out smoothly, keeping escape sequences atomic.
"""

from termflow.stream.smooth import (
    SmoothWriter,
    SteadyDrainer,
    StreamSmoother,
    split_by_visible,
)

__all__ = [
    "SmoothWriter",
    "SteadyDrainer",
    "StreamSmoother",
    "split_by_visible",
]
