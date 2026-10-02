"""The real DOOM (shareware episode) in a termflow window.

Run it::

    pip install "termflow-md[doom]"
    tf --doom-shareware
    python -m termflow.live.demos.doom_wasm --wad freedoom1.wad   # any IWAD/PWADs

The engine is `doom.wasm <https://github.com/jacobenget/doom.wasm>`_
(doomgeneric compiled to WebAssembly, GPL-2.0, shareware ``DOOM1.WAD``
built in), run by the ``wasmtime`` runtime. termflow does not ship it:
it is downloaded on first use from the project's GitHub release,
verified against a pinned SHA-256, and cached.

doom.wasm's whole interface is 10 imports and 4 exports, implemented by
:class:`DoomEngine`. Frames go through a letterboxed
:class:`~termflow.live.FramebufferView`; held keys are diffed into
Doom's key-down/key-up calls, so on terminals that report releases
(Windows Terminal, kitty protocol) it plays like the original.
No sound -- doom.wasm does not produce any.
"""

from __future__ import annotations

import argparse
import array
import ctypes
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from termflow.live.app import LiveApp, Window, hsplit
from termflow.live.demos.doom_sound import DoomSound, open_doom_sound
from termflow.live.input import ALT, CTRL, SHIFT
from termflow.live.widgets import FramebufferView, TextLog
from termflow.tui.keys import Key

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from termflow.live.buffer import Rect
    from termflow.live.input import KeyEvent, KeyState

T = TypeVar("T")

DOOM_WASM_VERSION = "v0.1.0"
DOOM_WASM_URL = (
    "https://github.com/jacobenget/doom.wasm/releases/download/"
    f"{DOOM_WASM_VERSION}/doom-{DOOM_WASM_VERSION}.wasm"
)
DOOM_WASM_SHA256 = "8edfe49a7583fd975199969302d8e9adcf8e714d0af72bf3e672f991fd810faa"
INSTALL_HINT = 'The real DOOM needs the wasmtime runtime: pip install "termflow-md[doom]"'

#: Classic DOOM renders 320x200 for a 4:3 CRT: non-square pixels.
DOOM_ASPECT = 4 / 3

#: Ticks a tap (pressed and released within one frame) stays down.
#: Doom samples key *state*, and a tick that runs no 35 Hz tic samples
#: nothing -- measured: 1 tick occasionally drops a tap, 2 never do.
TAP_TICKS = 2

#: termflow key name -> name of the doom.wasm global holding its doomKey.
_SPECIAL_KEYS = {
    Key.UP: "KEY_UPARROW",
    Key.DOWN: "KEY_DOWNARROW",
    Key.LEFT: "KEY_LEFTARROW",
    Key.RIGHT: "KEY_RIGHTARROW",
    Key.ENTER: "KEY_ENTER",
    Key.ESCAPE: "KEY_ESCAPE",
    Key.TAB: "KEY_TAB",
    Key.BACKSPACE: "KEY_BACKSPACE",
    CTRL: "KEY_FIRE",
    " ": "KEY_USE",
    SHIFT: "KEY_SHIFT",
    ALT: "KEY_ALT",
    ",": "KEY_STRAFE_L",
    ".": "KEY_STRAFE_R",
}


def doom_key(name: str, codes: dict[str, int]) -> int | None:
    """Map a termflow key name to a doomKey (None: Doom has no such key).

    Special keys use the constants exported by doom.wasm (``codes``);
    printable ASCII keys are their own character code, per the
    doom.wasm interface.
    """
    special = _SPECIAL_KEYS.get(name)
    if special is not None:
        return codes.get(special)
    if len(name) == 1 and " " < name < "\x7f":
        return ord(name)
    return None


def cache_dir() -> Path:
    """Per-user cache directory for downloaded engine files."""
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "termflow"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "termflow"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_doom_wasm(
    directory: Path | None = None,
    log: Callable[[str], None] = print,
    download: Callable[[str, Path], object] = urllib.request.urlretrieve,
) -> Path:
    """Return a verified local doom.wasm, downloading it once if needed.

    Raises:
        RuntimeError: When the download fails or its checksum is wrong.
    """
    directory = directory or cache_dir()
    target = directory / f"doom-{DOOM_WASM_VERSION}.wasm"
    if target.exists() and _sha256(target) == DOOM_WASM_SHA256:
        return target
    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    log(f"Downloading doom.wasm {DOOM_WASM_VERSION} (GPL-2.0, ~4.6 MB, includes the shareware WAD)")
    log(f"  from {DOOM_WASM_URL}")
    try:
        download(DOOM_WASM_URL, partial)
    except OSError as exc:
        raise RuntimeError(f"could not download doom.wasm: {exc}") from exc
    digest = _sha256(partial)
    if digest != DOOM_WASM_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"doom.wasm checksum mismatch (got {digest}); refusing to run it")
    partial.replace(target)
    log(f"  cached at {target}")
    return target


class DoomEngine:
    """doom.wasm hosted in wasmtime: implements its 10 imports.

    Args:
        wasm: Path to doom.wasm.
        wads: WAD files to load instead of the built-in shareware WAD.
        on_message: Receives Doom's console output, one line at a time.
    """

    def __init__(
        self,
        wasm: Path,
        wads: Sequence[Path] = (),
        on_message: Callable[[str], None] | None = None,
    ) -> None:
        try:
            import wasmtime
        except ImportError as exc:
            raise RuntimeError(INSTALL_HINT) from exc
        self._wads = [Path(p) for p in wads]
        self._on_message = on_message or (lambda _line: None)
        self._epoch = time.perf_counter()
        self.width = self.height = 0
        self._frame_ptr: int | None = None
        self.finished = False
        #: Optional sound (see :mod:`termflow.live.demos.doom_sound`).
        self.sound: DoomSound | None = None

        engine = wasmtime.Engine()
        self._store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        i32, i64 = wasmtime.ValType.i32(), wasmtime.ValType.i64()

        def define(module: str, name: str, params: list, results: list, fn: Any) -> None:
            linker.define_func(
                module, name, wasmtime.FuncType(params, results), fn, access_caller=True
            )

        define("loading", "onGameInit", [i32, i32], [], self._on_game_init)
        define("loading", "wadSizes", [i32, i32], [], self._wad_sizes)
        define("loading", "readWads", [i32, i32], [], self._read_wads)
        define("runtimeControl", "timeInMilliseconds", [], [i64], self._time_ms)
        define("ui", "drawFrame", [i32], [], self._draw_frame)
        define("gameSaving", "sizeOfSaveGame", [i32], [i32], lambda _c, _slot: 0)
        define("gameSaving", "readSaveGame", [i32, i32], [i32], lambda _c, _slot, _dst: 0)
        define("gameSaving", "writeSaveGame", [i32, i32, i32], [i32], lambda _c, _s, _p, _n: 0)
        define("console", "onInfoMessage", [i32, i32], [], self._message)
        define("console", "onErrorMessage", [i32, i32], [], self._message)

        instance = linker.instantiate(self._store, wasmtime.Module.from_file(engine, str(wasm)))
        exports = instance.exports(self._store)

        def export(name: str, kind: type[T]) -> T:
            item = exports[name]
            if not isinstance(item, kind):  # a downloaded binary: check its shape
                raise RuntimeError(
                    f"unexpected doom.wasm interface: {name} is not a {kind.__name__}"
                )
            return item

        self._memory = export("memory", wasmtime.Memory)
        self._tick = export("tickGame", wasmtime.Func)
        self._key_down = export("reportKeyDown", wasmtime.Func)
        self._key_up = export("reportKeyUp", wasmtime.Func)
        self.codes = {
            name: int(export(name, wasmtime.Global).value(self._store))
            for name in set(_SPECIAL_KEYS.values())
        }
        self._trap = (wasmtime.Trap, wasmtime.WasmtimeError)
        export("initGame", wasmtime.Func)(self._store)

    # -- imports ---------------------------------------------------------

    def _write_i32(self, caller: Any, offset: int, value: int) -> None:
        self._memory.write(caller, value.to_bytes(4, "little", signed=True), offset)

    def _on_game_init(self, _caller: Any, width: int, height: int) -> None:
        self.width, self.height = width, height

    def _wad_sizes(self, caller: Any, count_ptr: int, total_ptr: int) -> None:
        # Leaving the count at 0 means "load the built-in shareware WAD".
        if self._wads:
            self._write_i32(caller, count_ptr, len(self._wads))
            self._write_i32(caller, total_ptr, sum(p.stat().st_size for p in self._wads))

    def _read_wads(self, caller: Any, data_ptr: int, lengths_ptr: int) -> None:
        for i, path in enumerate(self._wads):
            data = path.read_bytes()
            self._memory.write(caller, data, data_ptr)
            self._write_i32(caller, lengths_ptr + 4 * i, len(data))
            data_ptr += len(data)

    def _time_ms(self, _caller: Any) -> int:
        # Real time. tickGame() busy-waits on this until the next 35 Hz
        # tic is due, which is exactly what paces the game. Doom also
        # calls it right after running each tic -- the one moment newly
        # started sounds are visible in memory -- so the sound watcher
        # polls here.
        if self.sound is not None:
            self.sound.watcher.poll(self.memory_base())
        return int((time.perf_counter() - self._epoch) * 1000)

    def _draw_frame(self, _caller: Any, ptr: int) -> None:
        self._frame_ptr = ptr  # copied once per tick in frame(), not per draw

    def _message(self, caller: Any, ptr: int, length: int) -> None:
        text = bytes(self._memory.read(caller, ptr, ptr + length)).decode("utf-8", "replace")
        for line in text.splitlines():
            if line.strip():
                self._on_message(line.rstrip())

    # -- driving -----------------------------------------------------------

    def tick(self) -> None:
        """Advance one game tic (blocks until it is due)."""
        if self.finished:
            return
        if self.sound is not None:
            self.sound.watcher.begin_tick()
        try:
            self._tick(self._store)
        except self._trap as exc:  # e.g. quitting from Doom's own menu
            self.finished = True
            self._on_message(f"DOOM exited: {str(exc).splitlines()[0]}")
        if self.sound is not None:
            self.sound.update_music(self.memory_base())

    def memory_base(self) -> int:
        """Host address of linear memory (re-read: memory can grow and move)."""
        return ctypes.addressof(self._memory.data_ptr(self._store).contents)

    def memory_snapshot(self) -> bytes:
        return ctypes.string_at(self.memory_base(), self._memory.data_len(self._store))

    def close(self) -> None:
        if self.sound is not None:
            self.sound.close()
            self.sound = None

    def key(self, name: str, pressed: bool) -> None:
        code = doom_key(name, self.codes)
        if code is not None and not self.finished:
            (self._key_down if pressed else self._key_up)(self._store, code)

    def frame(self) -> array.array[int] | None:
        """The latest frame as uint32 ``0x00RRGGBB`` pixels, row-major."""
        if self._frame_ptr is None:
            return None
        pixels = array.array("I")
        pixels.frombytes(
            ctypes.string_at(self.memory_base() + self._frame_ptr, self.width * self.height * 4)
        )
        return pixels


class DoomView(FramebufferView):
    """Plays a :class:`DoomEngine`: forwards held keys, shows frames."""

    def __init__(self, engine: DoomEngine, on_exit: Callable[[], None] | None = None) -> None:
        super().__init__(aspect=DOOM_ASPECT)
        self.engine = engine
        self.on_exit = on_exit or (lambda: None)
        self._down: frozenset[str] = frozenset()  # what Doom believes is held
        self._taps: dict[str, int] = {}  # tapped key -> ticks left to hold it

    def on_key(self, event: KeyEvent) -> bool:
        return event.key == Key.TAB  # Doom's automap, not window focus

    def update(self, dt: float, keys: KeyState) -> None:  # noqa: ARG002 - Doom keeps time
        for name in keys.pressed - keys.held:
            self._taps[name] = TAP_TICKS
        down = keys.held | frozenset(self._taps)
        for name in sorted(self._down - down):
            self.engine.key(name, pressed=False)
        for name in sorted(down - self._down):
            self.engine.key(name, pressed=True)
        self._down = down
        self.engine.tick()
        self._taps = {name: n - 1 for name, n in self._taps.items() if n > 1}
        if self.engine.finished:
            self.on_exit()
        frame = self.engine.frame()
        if frame is not None:
            self.set_frame(frame, self.engine.width, self.engine.height)


def doom_layout(area: Rect) -> list[Rect]:
    """Give DOOM a 4:3 window; the console log gets what is left."""
    doom_cols = min(area.width, round((area.height - 2) * 2 * DOOM_ASPECT) + 2)
    if area.width - doom_cols < 30:
        return [area]
    left, right = hsplit(area, doom_cols, area.width - doom_cols)
    return [left, right]


def build_app(wasm: Path, wads: Sequence[Path] = (), sound: bool = True) -> LiveApp:
    log = TextLog()

    def to_log(line: str) -> None:
        log.write(line + "\n")

    engine = DoomEngine(wasm, wads, on_message=to_log)
    if sound:
        engine.sound = open_doom_sound(wasm, wads, engine.memory_snapshot(), to_log)
    view = DoomView(engine)
    app = LiveApp(
        [Window("DOOM", view), Window("Console", log)],
        doom_layout,
        title="termflow × DOOM",  # noqa: RUF001 - a multiplication sign, on purpose
        # Doom simulates at 35 Hz and tickGame() returns early when no tic
        # is due; polling a bit faster never misses one, and duplicate
        # frames diff to (almost) zero bytes.
        fps=40,
        hints="Ctrl+Q quit",
    )
    view.on_exit = app.quit

    def status() -> str:
        held = "native key-up" if app.console.reports_release else "autorepeat"
        return f"arrows move · Ctrl fire · Space use · Esc menu · Tab map · input: {held}"

    app.status = status
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play DOOM (doom.wasm) in a termflow window")
    parser.add_argument(
        "--wad", type=Path, action="append", default=[], help="WAD to load (repeatable)"
    )
    parser.add_argument(
        "--wasm", type=Path, default=None, help="use this doom.wasm instead of downloading"
    )
    parser.add_argument("--no-sound", action="store_true", help="no effects or music")
    args = parser.parse_args(argv)
    try:
        wasm = args.wasm or fetch_doom_wasm()
        app = build_app(wasm, args.wad, sound=not args.no_sound)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    view = app.windows[0].widget
    try:
        stats = app.run()
    finally:
        if isinstance(view, DoomView):
            view.engine.close()  # stop the audio threads
    print(f"{stats.frames} frames in {stats.seconds:.1f}s ({stats.fps:.1f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
