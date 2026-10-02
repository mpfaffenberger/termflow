"""Real-time keyboard input: press/release events and held-key state.

Menus only care that a key was pressed. Games care whether it is
*still held* -- which most terminals never tell you. Two sources do:

* **Windows** consoles (Windows Terminal, conhost) deliver true key-up
  records through ``ReadConsoleInputW`` (see :mod:`termflow.live._win32`).
* The **kitty keyboard protocol** (kitty, WezTerm, foot, Ghostty,
  recent iTerm2/Alacritty...) reports release events as ``CSI ... u``.
  :class:`VTInputParser` speaks it, plus the legacy sequences.

Everywhere else :class:`KeyState` fakes holds from autorepeat: a press
counts as held for a moment and every repeat extends it. Slightly
sticky, but playable.

Key names match :class:`termflow.tui.keys.Key`; letters are always
lowercase (``"w"`` even with shift held), ctrl combos are ``"ctrl-x"``,
and the modifiers themselves are ``"shift"`` / ``"ctrl"`` / ``"alt"``
when the backend reports them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from termflow.tui.keys import Key, parse_key

SHIFT = "shift"
CTRL = "ctrl"
ALT = "alt"
_CTRL_PREFIX = "ctrl-"


@dataclass(frozen=True)
class KeyEvent:
    """One key transition.

    Attributes:
        key: Normalized key name (see module docs).
        pressed: True for press/repeat, False for release.
        repeat: True for autorepeat presses (when the backend knows).
        text: The text the key types, if any (``"W"`` for shift-w).
    """

    key: str
    pressed: bool = True
    repeat: bool = False
    text: str = ""


def base_key(key: str) -> str:
    """Strip a ``ctrl-`` prefix: the physical key behind a combo."""
    return key[len(_CTRL_PREFIX) :] if key.startswith(_CTRL_PREFIX) and len(key) > 5 else key


class KeyState:
    """Which keys are held right now, and which went down this frame.

    Feed it events with :meth:`feed`; call :meth:`end_frame` once per
    frame after game logic ran. Queries take any number of key names
    and answer "any of these" (handy for WASD + arrow bindings).
    """

    def __init__(self, initial_hold: float = 0.55, repeat_hold: float = 0.12) -> None:
        self.initial_hold = initial_hold
        self.repeat_hold = repeat_hold
        self._down: dict[str, float | None] = {}  # key -> expiry (None: until release)
        self._pressed: set[str] = set()

    def feed(self, event: KeyEvent, now: float, reports_release: bool) -> None:
        key = base_key(event.key)
        if not event.pressed:
            self._down.pop(key, None)
            return
        held = self.is_down(key)
        if not held:
            self._pressed.add(key)
        if reports_release:
            self._down[key] = None
        else:
            self._down[key] = now + (self.repeat_hold if held else self.initial_hold)

    def expire(self, now: float) -> None:
        """Drop synthetic holds whose autorepeat went quiet."""
        stale = [k for k, until in self._down.items() if until is not None and until <= now]
        for k in stale:
            del self._down[k]

    def end_frame(self) -> None:
        self._pressed.clear()

    def release_all(self) -> None:
        """Forget everything (focus changes, so keys never get stuck)."""
        self._down.clear()
        self._pressed.clear()

    @property
    def held(self) -> frozenset[str]:
        """Every key currently down (diff two of these for press/release)."""
        return frozenset(self._down)

    @property
    def pressed(self) -> frozenset[str]:
        """Keys that went down this frame -- including taps already released."""
        return frozenset(self._pressed)

    def is_down(self, *keys: str) -> bool:
        return any(k in self._down for k in keys)

    def was_pressed(self, *keys: str) -> bool:
        return any(k in self._pressed for k in keys)

    def axis(self, negative: tuple[str, ...], positive: tuple[str, ...]) -> int:
        """-1, 0, or +1 from two opposing key groups."""
        return int(self.is_down(*positive)) - int(self.is_down(*negative))


#: Kitty / CSI-u key codes that are not their own character.
_CSI_U_CODES = {
    9: Key.TAB,
    13: Key.ENTER,
    27: Key.ESCAPE,
    127: Key.BACKSPACE,
    57441: SHIFT,
    57447: SHIFT,
    57442: CTRL,
    57448: CTRL,
    57443: ALT,
    57449: ALT,
}

#: Final byte (or ``N~``) of legacy/functional CSI sequences.
_CSI_FINALS = {
    "A": Key.UP,
    "B": Key.DOWN,
    "C": Key.RIGHT,
    "D": Key.LEFT,
    "H": Key.HOME,
    "F": Key.END,
    "P": "f1",
    "Q": "f2",
    "R": "f3",
    "S": "f4",
}
_CSI_TILDE = {
    1: Key.HOME,
    3: Key.DELETE,
    4: Key.END,
    5: Key.PAGE_UP,
    6: Key.PAGE_DOWN,
    7: Key.HOME,
    8: Key.END,
}

_MOD_SHIFT, _MOD_CTRL = 1, 4

# CSI: ESC [ params final ; SS3: ESC O x
_CSI_RE = re.compile(r"\x1b\[([0-9;:?<=>]*)([ -/]*)([@-~])")


class VTInputParser:
    """Incremental decoder for terminal key input (legacy + kitty CSI-u).

    Feed raw text with :meth:`feed`; an incomplete trailing escape
    sequence is held back until more input arrives or :meth:`flush` is
    called (a lone ``ESC`` is only a keypress once the burst is over).
    """

    def __init__(self) -> None:
        self._pending = ""
        #: Set when the terminal answers the kitty protocol query.
        self.kitty = False

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    def flush(self) -> list[KeyEvent]:
        """Resolve held-back input (call when the input burst is over)."""
        data, self._pending = self._pending, ""
        events = []
        for ch in data:
            key = parse_key(ch)
            if key is not None:
                events.append(_char_event(key, ch))
        return events

    def feed(self, data: str) -> list[KeyEvent]:
        data = self._pending + data
        self._pending = ""
        events: list[KeyEvent] = []
        i = 0
        while i < len(data):
            ch = data[i]
            if ch != "\x1b":
                key = parse_key(ch)
                if key is not None:
                    events.append(_char_event(key, ch))
                i += 1
                continue
            rest = data[i:]
            if len(rest) == 1 or (len(rest) == 2 and rest[1] in "[O"):
                self._pending = rest  # maybe the start of a sequence
                break
            if rest[1] == "O":  # SS3: ESC O x
                name = _CSI_FINALS.get(rest[2])
                if name:
                    events.append(KeyEvent(name))
                i += 3
                continue
            if rest[1] == "[":
                m = _CSI_RE.match(rest)
                if m is None:
                    if len(rest) < 32:
                        self._pending = rest  # incomplete CSI
                        break
                    i += 2  # garbage: skip the introducer
                    continue
                event = self._decode_csi(m.group(1), m.group(3))
                if event is not None:
                    events.append(event)
                i += m.end()
                continue
            # ESC + char: alt-modified key. Report the key itself.
            key = parse_key(rest[1])
            if key is not None:
                events.append(_char_event(key, rest[1]))
            i += 2
        return events

    def _decode_csi(self, params: str, final: str) -> KeyEvent | None:
        if params.startswith("?"):
            if final == "u":
                self.kitty = True  # reply to our "CSI ? u" query
            return None
        fields = params.split(";")
        if final == "u":
            code_part = fields[0].split(":")[0]
            if not code_part.isdigit():
                return None
            name, text = _codepoint_key(int(code_part))
        elif final == "~":
            num = fields[0]
            name = _CSI_TILDE.get(int(num)) if num.isdigit() else None
            text = ""
        else:
            name, text = _CSI_FINALS.get(final), ""
        if name is None:
            return None
        mods, event_type = _parse_modifiers(fields[1] if len(fields) > 1 else "")
        if mods & _MOD_CTRL and len(name) == 1 and name.isalpha():
            name, text = _CTRL_PREFIX + name, ""
        elif mods & _MOD_SHIFT and text:
            text = text.upper()
        return KeyEvent(name, pressed=event_type != 3, repeat=event_type == 2, text=text)


def _char_event(key: str, ch: str) -> KeyEvent:
    printable = len(key) == 1
    return KeyEvent(key.lower() if printable else key, text=ch if printable else "")


def _codepoint_key(code: int) -> tuple[str | None, str]:
    if code in _CSI_U_CODES:
        return _CSI_U_CODES[code], ""
    if 32 <= code < 0x10FFFF and not 57344 <= code <= 63743:  # skip private-use keys
        ch = chr(code)
        return ch.lower(), ch
    return None, ""


def _parse_modifiers(field: str) -> tuple[int, int]:
    """``mods[:event]`` -> (modifier bitmask, event type 1/2/3)."""
    if not field:
        return 0, 1
    mods, _, event = field.partition(":")
    bits = int(mods) - 1 if mods.isdigit() and int(mods) > 0 else 0
    return bits, int(event) if event.isdigit() else 1
