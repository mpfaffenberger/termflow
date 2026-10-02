"""Sound effects and music for doom.wasm -- which has no sound output at all.

doom.wasm's interface has no audio imports, but Doom still runs its own
sound bookkeeping internally, in linear memory we can read:

* ``S_sfx[]`` (109 x 48-byte ``sfxinfo_t``): ``S_StartSound`` bumps an
  entry's ``usefulness``, and ``S_UpdateSounds`` drops it again later
  in the *same* loop iteration (no sound driver means "never playing").
  The game calls our ``timeInMilliseconds`` import right after each tic
  runs (``NetUpdate`` -> ``I_GetTime``), so polling there catches every
  sound start.
* ``S_music[]`` (68 x 16-byte ``musicinfo_t``, right before ``S_sfx``):
  the entry whose ``data`` pointer is set is the song playing now.

Samples come from the WAD (the shareware IWAD embedded in doom.wasm, or
the user's WADs): ``DS*`` lumps are 8-bit DMX PCM, ``D_*`` lumps are MUS
scores, played as MIDI through the system synth. Windows only for now
(:mod:`termflow.live._winmm`); elsewhere :func:`open_doom_sound` returns
None. Effects play at full volume: positional volume/panning is
computed inside Doom and never reaches memory we can see.
"""

from __future__ import annotations

import array
import ctypes
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

SFX_STRIDE, SFX_NAME_OFFSET, SFX_USEFULNESS_OFFSET = 48, 4, 32
SFX_LAST = b"radio"  # sfx_radio: the final S_sfx entry
MUSIC_STRIDE, MUSIC_COUNT = 16, 68  # NUMMUSIC; entry 0 is mus_None
MUS_TICKS_PER_SECOND = 140
MIXER_RATE = 11025  # Doom's native effect rate
SFX_GAIN = 160  # unsigned 8-bit -> signed 16-bit, with headroom for 8 voices


# -- WAD + lump formats -------------------------------------------------------


def read_wad(data: bytes, offset: int = 0) -> dict[str, bytes]:
    """Lump name -> bytes for the WAD starting at ``offset`` in ``data``."""
    magic = data[offset : offset + 4]
    if magic not in (b"IWAD", b"PWAD"):
        raise ValueError("not a WAD")
    count, directory = struct.unpack_from("<ii", data, offset + 4)
    lumps = {}
    for i in range(count):
        pos, size, raw = struct.unpack_from("<ii8s", data, offset + directory + 16 * i)
        lumps[raw.rstrip(b"\0").decode("ascii", "replace").upper()] = data[
            offset + pos : offset + pos + size
        ]
    return lumps


def find_embedded_iwad(blob: bytes) -> dict[str, bytes]:
    """The IWAD baked into a binary (doom.wasm carries DOOM1.WAD)."""
    pos = blob.find(b"IWAD")
    while pos >= 0:
        try:
            lumps = read_wad(blob, pos)
            if "PLAYPAL" in lumps:
                return lumps
        except (ValueError, struct.error):
            pass
        pos = blob.find(b"IWAD", pos + 1)
    raise ValueError("no embedded IWAD found")


def decode_dmx(lump: bytes) -> tuple[int, array.array[int]]:
    """DMX sound lump -> (sample rate, signed 16-bit samples).

    Layout: u16 format (3), u16 rate, u32 count, then ``count`` unsigned
    8-bit samples of which the first and last 16 are padding.
    """
    fmt, rate, count = struct.unpack_from("<HHI", lump)
    if fmt != 3:
        raise ValueError("not a DMX sound")
    pcm = lump[8 + 16 : 8 + count - 16]
    return rate, array.array("h", [(s - 128) * SFX_GAIN for s in pcm])


def resample(rate: int, samples: array.array[int], target: int) -> array.array[int]:
    """Nearest-neighbor rate conversion (identity when rates match)."""
    if rate == target or not samples:
        return samples
    count = len(samples) * target // rate
    return array.array("h", [samples[i * rate // target] for i in range(count)])


#: MUS controller number -> MIDI controller (0 is program change).
_MUS_CONTROLLERS = {1: 0, 2: 1, 3: 7, 4: 10, 5: 11, 6: 91, 7: 93, 8: 64, 9: 67}
#: MUS system events -> MIDI channel-mode controllers.
_MUS_SYSTEM = {10: 120, 11: 123, 12: 126, 13: 127, 14: 121}


def mus_to_midi(lump: bytes) -> list[tuple[int, int, int, int]]:
    """MUS score -> ``(tick, status, data1, data2)`` MIDI events (140 Hz ticks)."""
    if lump[:4] != b"MUS\x1a":
        raise ValueError("not a MUS lump")
    length, start = struct.unpack_from("<HH", lump, 4)
    pos, end, tick = start, start + length, 0
    velocity = [127] * 16
    events: list[tuple[int, int, int, int]] = []
    while pos < end:
        byte = lump[pos]
        pos += 1
        kind, mus_channel, last = byte >> 4 & 7, byte & 15, byte & 0x80
        channel = 9 if mus_channel == 15 else mus_channel + 1 if mus_channel >= 9 else mus_channel
        if kind == 0:  # release note
            events.append((tick, 0x80 | channel, lump[pos] & 127, 0))
            pos += 1
        elif kind == 1:  # play note (optional new velocity)
            note = lump[pos]
            pos += 1
            if note & 0x80:
                velocity[channel] = lump[pos] & 127
                pos += 1
            events.append((tick, 0x90 | channel, note & 127, velocity[channel]))
        elif kind == 2:  # pitch bend: 0..255, 128 = center
            bend = lump[pos] * 64
            pos += 1
            events.append((tick, 0xE0 | channel, bend & 127, bend >> 7 & 127))
        elif kind == 3:  # system event
            controller = _MUS_SYSTEM.get(lump[pos])
            pos += 1
            if controller is not None:
                events.append((tick, 0xB0 | channel, controller, 0))
        elif kind == 4:  # controller change
            number, value = lump[pos], lump[pos + 1] & 127
            pos += 2
            if number == 0:
                events.append((tick, 0xC0 | channel, value, 0))
            elif number in _MUS_CONTROLLERS:
                events.append((tick, 0xB0 | channel, _MUS_CONTROLLERS[number], value))
        elif kind == 6:  # score end
            break
        if last:  # variable-length delay follows
            delay = 0
            while True:
                b = lump[pos]
                pos += 1
                delay = delay << 7 | b & 127
                if not b & 0x80:
                    break
            tick += delay
    return events


# -- watching Doom's memory -------------------------------------------------------


@dataclass(frozen=True)
class SoundTables:
    """Where S_sfx and S_music live in doom.wasm's linear memory."""

    sfx_start: int
    sfx_names: tuple[str, ...]
    music_start: int
    music_names: tuple[str, ...]

    @classmethod
    def locate(cls, memory: bytes) -> SoundTables:
        """Find both tables by their names; raises ValueError if absent."""
        pistol = memory.find(b"\0pistol\0") + 1  # S_sfx[1]
        if pistol <= 0:
            raise ValueError("sound table not found")
        sfx_start = pistol - SFX_NAME_OFFSET - SFX_STRIDE
        names = []
        for i in range(256):
            off = sfx_start + i * SFX_STRIDE + SFX_NAME_OFFSET
            raw = memory[off : off + 9].split(b"\0")[0]
            names.append(raw.decode("ascii", "replace"))
            if raw == SFX_LAST:
                break
        else:
            raise ValueError("sound table end not found")
        music_start = sfx_start - MUSIC_COUNT * MUSIC_STRIDE
        music = []
        for i in range(MUSIC_COUNT):
            ptr = struct.unpack_from("<i", memory, music_start + i * MUSIC_STRIDE)[0]
            music.append(
                memory[ptr : ptr + 9].split(b"\0")[0].decode("ascii", "replace") if i else ""
            )
        if music[1] != "e1m1":
            raise ValueError("music table not found")
        return cls(sfx_start, tuple(names), music_start, tuple(music))

    def active_sfx(self, base: int) -> list[int]:
        """``usefulness`` of every sound (live view into memory at ``base``)."""
        words = SFX_STRIDE // 4
        view = (ctypes.c_int32 * (len(self.sfx_names) * words)).from_address(base + self.sfx_start)
        return view[SFX_USEFULNESS_OFFSET // 4 :: words]

    def current_song(self, base: int) -> str | None:
        view = (ctypes.c_int32 * (MUSIC_COUNT * 4)).from_address(base + self.music_start)
        for i in range(1, MUSIC_COUNT):
            if view[4 * i + 2]:  # musicinfo_t.data
                return self.music_names[i]
        return None


class SoundWatcher:
    """Turns S_sfx bookkeeping into "sound started" events.

    Call :meth:`begin_tick` before each ``tickGame()`` and :meth:`poll`
    from the time import. A sound's ``usefulness`` counts channels it
    started on since the last ``S_UpdateSounds``; any rise is a start.
    """

    def __init__(self, tables: SoundTables, on_sound: Callable[[str], None]) -> None:
        self.tables = tables
        self.on_sound = on_sound
        self._seen = [0] * len(tables.sfx_names)

    def begin_tick(self) -> None:
        self._seen = [0] * len(self._seen)

    def poll(self, base: int) -> None:
        for i, value in enumerate(self.tables.active_sfx(base)):
            if value > self._seen[i]:
                for _ in range(value - self._seen[i]):
                    self.on_sound(self.tables.sfx_names[i])
                self._seen[i] = value


# -- playback ---------------------------------------------------------------------


class MusicPlayer:  # pragma: no cover - needs a MIDI device
    """Loops one MIDI event list on a background thread."""

    def __init__(self, midi: Any) -> None:
        self._midi = midi
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def play(self, events: Sequence[tuple[int, int, int, int]]) -> None:
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(events,), name="termflow-mus", daemon=True
        )
        self._thread.start()

    def _run(self, events: Sequence[tuple[int, int, int, int]]) -> None:
        while not self._stop.is_set():  # Doom loops level music
            start = time.perf_counter()
            for tick, status, d1, d2 in events:
                delay = start + tick / MUS_TICKS_PER_SECOND - time.perf_counter()
                if delay > 0 and self._stop.wait(delay):
                    return
                self._midi.send(status, d1, d2)
            self._midi.reset()  # silence hanging notes before the loop restarts

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self._midi.reset()


class DoomSound:  # pragma: no cover - needs audio devices
    """Plays Doom's sound effects and music from its WAD data."""

    def __init__(self, lumps: dict[str, bytes], tables: SoundTables, mixer: Any, midi: Any) -> None:
        self._lumps, self._mixer, self._midi = lumps, mixer, midi
        self._clips: dict[str, array.array[int] | None] = {}
        self._music = MusicPlayer(midi) if midi is not None else None
        self._song: str | None = None
        self.tables = tables
        self.watcher = SoundWatcher(tables, self._play_sfx)

    def _play_sfx(self, name: str) -> None:
        if name not in self._clips:
            lump = self._lumps.get(f"DS{name.upper()}")
            try:
                self._clips[name] = resample(*decode_dmx(lump), MIXER_RATE) if lump else None
            except (ValueError, struct.error):
                self._clips[name] = None
        clip = self._clips[name]
        if clip is not None and self._mixer is not None:
            self._mixer.play(clip)

    def update_music(self, base: int) -> None:
        song = self.tables.current_song(base)
        if song == self._song or self._music is None:
            return
        self._song = song
        lump = self._lumps.get(f"D_{song.upper()}") if song else None
        if lump is None:
            self._music.stop()
            return
        try:
            self._music.play(mus_to_midi(lump))
        except (ValueError, IndexError, struct.error):
            self._music.stop()

    def close(self) -> None:
        if self._music is not None:
            self._music.stop()
        for device in (self._mixer, self._midi):
            if device is not None:
                device.close()


def load_lumps(wasm: Path, wads: Sequence[Path]) -> dict[str, bytes]:
    """The lumps Doom itself loads: the user's WADs in order, else the embedded IWAD."""
    if not wads:
        return find_embedded_iwad(wasm.read_bytes())
    lumps: dict[str, bytes] = {}
    for path in wads:
        lumps.update(read_wad(path.read_bytes()))  # later WADs override earlier ones
    return lumps


def open_doom_sound(
    wasm: Path, wads: Sequence[Path], memory: bytes, log: Callable[[str], None]
) -> DoomSound | None:  # pragma: no cover - platform glue
    """Sound for this platform and engine, or None (with a reason logged)."""
    if sys.platform != "win32":
        log("sound: only supported on Windows so far")
        return None
    from termflow.live._winmm import MidiOut, WaveMixer

    try:
        tables = SoundTables.locate(memory)
        lumps = load_lumps(wasm, wads)
    except (ValueError, OSError) as exc:
        log(f"sound: disabled ({exc})")
        return None
    mixer = midi = None
    try:
        mixer = WaveMixer(rate=MIXER_RATE)
    except OSError as exc:
        log(f"sound: no effects ({exc})")
    try:
        midi = MidiOut()
    except OSError as exc:
        log(f"sound: no music ({exc})")
    if mixer is None and midi is None:
        return None
    log(f"sound: {len(tables.sfx_names)} effects, music via the system MIDI synth")
    return DoomSound(lumps, tables, mixer, midi)
