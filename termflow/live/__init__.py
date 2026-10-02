"""Live mode: real-time, multi-window terminal apps (yes, including games).

Where :mod:`termflow.tui` blocks on one key at a time, live mode runs a
frame loop: non-blocking input with true key-up events, a cell buffer
diffed against the previous frame, and windows that can host markdown,
logs, or raw RGB framebuffers drawn with half-block pixels.

Works in Windows Terminal (and conhost) via native console input
records, and in POSIX terminals via raw mode plus the kitty keyboard
protocol where available. Try it: ``tf --doom``.
"""

from termflow.live.app import AppStats, LiveApp, Widget, Window, hsplit, vsplit
from termflow.live.buffer import DEFAULT, Rect, Region, ScreenBuffer, render_diff, rgb
from termflow.live.console import LiveConsole
from termflow.live.input import KeyEvent, KeyState, VTInputParser
from termflow.live.pixels import PixelSurface
from termflow.live.widgets import FramebufferView, MarkdownView, PixelWidget, TextLog

__all__ = [
    "DEFAULT",
    "AppStats",
    "FramebufferView",
    "KeyEvent",
    "KeyState",
    "LiveApp",
    "LiveConsole",
    "MarkdownView",
    "PixelSurface",
    "PixelWidget",
    "Rect",
    "Region",
    "ScreenBuffer",
    "TextLog",
    "VTInputParser",
    "Widget",
    "Window",
    "hsplit",
    "render_diff",
    "rgb",
    "vsplit",
]
