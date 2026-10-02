"""Windows console backend: VT output + native key press/release input.

Windows Terminal (via ConPTY) and classic conhost both deliver full
``KEY_EVENT_RECORD`` s -- including key-*up* -- to
``ReadConsoleInputW``. That beats any VT input mode for games, so live
mode reads console records directly instead of parsing escape codes.

Handles are opened on ``CONIN$`` / ``CONOUT$`` so this works even when
stdin/stdout are redirected. :func:`translate_key` is pure (testable on
any OS); ctypes is only touched by :class:`WindowsConsoleIO`.
"""

from __future__ import annotations

import ctypes
from typing import Any

from termflow.live.input import ALT, CTRL, SHIFT, KeyEvent
from termflow.tui.keys import Key

_VK_NAMES = {
    0x08: Key.BACKSPACE,
    0x09: Key.TAB,
    0x0D: Key.ENTER,
    0x10: SHIFT,
    0x11: CTRL,
    0x12: ALT,
    0x1B: Key.ESCAPE,
    0x20: " ",
    0x21: Key.PAGE_UP,
    0x22: Key.PAGE_DOWN,
    0x23: Key.END,
    0x24: Key.HOME,
    0x25: Key.LEFT,
    0x26: Key.UP,
    0x27: Key.RIGHT,
    0x28: Key.DOWN,
    0x2E: Key.DELETE,
    **{0x70 + n: f"f{n + 1}" for n in range(12)},
}

_RIGHT_ALT, _LEFT_ALT, _RIGHT_CTRL, _LEFT_CTRL = 0x1, 0x2, 0x4, 0x8

# Console mode flags.
_ENABLE_PROCESSED_OUTPUT = 0x0001
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
_ENABLE_WINDOW_INPUT = 0x0008
_ENABLE_EXTENDED_FLAGS = 0x0080  # set without QUICK_EDIT: no accidental selection pauses
_KEY_EVENT = 0x0001

_GENERIC_READ_WRITE = 0x80000000 | 0x40000000
_FILE_SHARE_READ_WRITE = 0x1 | 0x2
_OPEN_EXISTING = 3
_INVALID_HANDLE = ctypes.c_void_p(-1).value


def translate_key(vk: int, key_down: bool, char: str, control_state: int) -> KeyEvent | None:
    """Map one console key record to a :class:`KeyEvent` (None: ignore)."""
    ctrl = bool(control_state & (_LEFT_CTRL | _RIGHT_CTRL))
    alt = bool(control_state & (_LEFT_ALT | _RIGHT_ALT))
    if 0x41 <= vk <= 0x5A:  # letters: name by the physical key
        letter = chr(vk).lower()
        if ctrl and not alt:  # ctrl+alt is AltGr on many layouts: keep the text
            return KeyEvent(f"ctrl-{letter}", pressed=key_down)
        return KeyEvent(letter, pressed=key_down, text=char if char.isprintable() else "")
    name = _VK_NAMES.get(vk)
    if name is not None:
        return KeyEvent(name, pressed=key_down, text=" " if name == " " else "")
    if char and char.isprintable():  # digits, punctuation, layout-specific keys
        return KeyEvent(char.lower(), pressed=key_down, text=char)
    return None


class _KeyEventRecord(ctypes.Structure):
    _fields_ = [
        ("bKeyDown", ctypes.c_int),
        ("wRepeatCount", ctypes.c_ushort),
        ("wVirtualKeyCode", ctypes.c_ushort),
        ("wVirtualScanCode", ctypes.c_ushort),
        ("UnicodeChar", ctypes.c_wchar),
        ("dwControlKeyState", ctypes.c_uint32),
    ]


class _EventUnion(ctypes.Union):
    _fields_ = [("KeyEvent", _KeyEventRecord), ("_pad", ctypes.c_byte * 16)]


class _InputRecord(ctypes.Structure):
    _fields_ = [("EventType", ctypes.c_ushort), ("Event", _EventUnion)]


class WindowsConsoleIO:  # pragma: no cover - needs a real Windows console
    """Puts the console into game mode and polls key records."""

    reports_release = True

    def __init__(self) -> None:
        self._k32: Any = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        self._k32.CreateFileW.restype = ctypes.c_void_p
        self._k32.CreateFileW.argtypes = [
            ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ]  # fmt: skip
        self._in = self._out = 0  # opened per session in start()
        self._saved: list[tuple[int, int]] = []
        self._records = (_InputRecord * 64)()

    def _open(self, name: str) -> int:
        handle = self._k32.CreateFileW(
            name, _GENERIC_READ_WRITE, _FILE_SHARE_READ_WRITE, None, _OPEN_EXISTING, 0, None
        )
        if handle in (None, _INVALID_HANDLE):
            raise OSError(f"live mode needs a console: cannot open {name}")
        return int(handle)

    def _get_mode(self, handle: int) -> int:
        mode = ctypes.c_uint32()
        if not self._k32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)):
            raise OSError("GetConsoleMode failed (not a console?)")
        return mode.value

    def _set_mode(self, handle: int, mode: int) -> None:
        self._k32.SetConsoleMode(ctypes.c_void_p(handle), ctypes.c_uint32(mode))

    def start(self) -> None:
        self._in, self._out = self._open("CONIN$"), self._open("CONOUT$")
        try:
            in_mode, out_mode = self._get_mode(self._in), self._get_mode(self._out)
        except OSError:
            self.stop()  # close the handles we just opened
            raise
        self._saved = [(self._in, in_mode), (self._out, out_mode)]
        # No line/echo/processed input: every key (ctrl-c included) is a record.
        self._set_mode(self._in, _ENABLE_WINDOW_INPUT | _ENABLE_EXTENDED_FLAGS)
        self._set_mode(
            self._out, out_mode | _ENABLE_PROCESSED_OUTPUT | _ENABLE_VIRTUAL_TERMINAL_PROCESSING
        )
        self._k32.FlushConsoleInputBuffer(ctypes.c_void_p(self._in))

    def stop(self) -> None:
        for handle, mode in self._saved:
            self._set_mode(handle, mode)
        self._saved = []
        for handle in (self._in, self._out):
            self._k32.CloseHandle(ctypes.c_void_p(handle))
        self._in = self._out = 0

    def poll(self) -> list[KeyEvent]:
        """All pending key events, without blocking."""
        events: list[KeyEvent] = []
        count, read = ctypes.c_uint32(), ctypes.c_uint32()
        handle = ctypes.c_void_p(self._in)
        while self._k32.GetNumberOfConsoleInputEvents(handle, ctypes.byref(count)) and count.value:
            n = min(count.value, len(self._records))
            if not self._k32.ReadConsoleInputW(handle, self._records, n, ctypes.byref(read)):
                break
            for rec in self._records[: read.value]:
                if rec.EventType != _KEY_EVENT:
                    continue  # resize/focus/menu records: size is polled instead
                k = rec.Event.KeyEvent
                event = translate_key(
                    k.wVirtualKeyCode, bool(k.bKeyDown), k.UnicodeChar, k.dwControlKeyState
                )
                if event is not None:
                    events.append(event)
        return events
