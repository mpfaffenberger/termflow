"""Windows audio output via winmm (ctypes): a PCM mixer and a MIDI port.

No dependencies: ``waveOut`` streams mixed 16-bit mono PCM from a
background thread, ``midiOut`` sends short MIDI messages to the built-in
Microsoft GS Wavetable Synth. Both no-op cleanly if no device opens.
"""

from __future__ import annotations

import array
import ctypes
import threading
import time
from typing import Any

_WAVE_MAPPER = 0xFFFFFFFF
_WHDR_DONE = 0x1
_MMSYSERR_NOERROR = 0


class _WaveFormat(ctypes.Structure):
    _fields_ = [
        ("wFormatTag", ctypes.c_ushort),
        ("nChannels", ctypes.c_ushort),
        ("nSamplesPerSec", ctypes.c_uint32),
        ("nAvgBytesPerSec", ctypes.c_uint32),
        ("nBlockAlign", ctypes.c_ushort),
        ("wBitsPerSample", ctypes.c_ushort),
        ("cbSize", ctypes.c_ushort),
    ]


class _WaveHeader(ctypes.Structure):
    _fields_ = [
        ("lpData", ctypes.c_void_p),
        ("dwBufferLength", ctypes.c_uint32),
        ("dwBytesRecorded", ctypes.c_uint32),
        ("dwUser", ctypes.c_size_t),
        ("dwFlags", ctypes.c_uint32),
        ("dwLoops", ctypes.c_uint32),
        ("lpNext", ctypes.c_void_p),
        ("reserved", ctypes.c_size_t),
    ]


def _winmm() -> Any:
    return ctypes.WinDLL("winmm")  # type: ignore[attr-defined]


class WaveMixer:  # pragma: no cover - needs a Windows audio device
    """Mixes short 16-bit mono clips into a continuous waveOut stream.

    Args:
        rate: Sample rate of every clip (and the device stream).
        voices: Max simultaneous clips; the oldest is dropped beyond it.
        chunk: Samples per buffer (latency ~= buffers * chunk / rate).
    """

    def __init__(
        self, rate: int = 11025, voices: int = 8, chunk: int = 256, buffers: int = 4
    ) -> None:
        self._mm = _winmm()
        fmt = _WaveFormat(1, 1, rate, rate * 2, 2, 16, 0)
        self._handle = ctypes.c_void_p()
        if self._mm.waveOutOpen(
            ctypes.byref(self._handle), ctypes.c_uint(_WAVE_MAPPER), ctypes.byref(fmt), 0, 0, 0
        ):
            raise OSError("no audio output device (waveOutOpen failed)")
        self._chunk, self._max_voices = chunk, voices
        self._voices: list[list[Any]] = []  # [samples, position]
        self._lock = threading.Lock()
        self._bufs = [ctypes.create_string_buffer(chunk * 2) for _ in range(buffers)]
        self._hdrs = [
            _WaveHeader(ctypes.cast(b, ctypes.c_void_p), chunk * 2, 0, 0, 0, 0, None, 0)
            for b in self._bufs
        ]
        for hdr in self._hdrs:
            self._mm.waveOutPrepareHeader(self._handle, ctypes.byref(hdr), ctypes.sizeof(hdr))
            hdr.dwFlags |= _WHDR_DONE  # "free": ready to be filled
        self.buffers_played = 0
        self._running = True
        self._thread = threading.Thread(target=self._pump, name="termflow-wave", daemon=True)
        self._thread.start()

    def play(self, samples: array.array[int]) -> None:
        """Start a clip (signed 16-bit samples at the mixer rate)."""
        with self._lock:
            self._voices.append([samples, 0])
            if len(self._voices) > self._max_voices:
                self._voices.pop(0)

    def _mix(self) -> bytes:
        n = self._chunk
        mix = [0] * n
        with self._lock:
            alive = []
            for voice in self._voices:
                samples, pos = voice
                part = samples[pos : pos + n]
                # A clip's last chunk is short: zip stops at its end.
                mix[: len(part)] = [a + b for a, b in zip(mix, part, strict=False)]
                voice[1] = pos + n
                if voice[1] < len(samples):
                    alive.append(voice)
            self._voices = alive
        return array.array("h", [max(-32768, min(32767, v)) for v in mix]).tobytes()

    def _pump(self) -> None:
        while self._running:
            for buf, hdr in zip(self._bufs, self._hdrs, strict=True):
                if hdr.dwFlags & _WHDR_DONE:
                    data = self._mix()
                    ctypes.memmove(buf, data, len(data))
                    hdr.dwFlags &= ~_WHDR_DONE
                    self._mm.waveOutWrite(self._handle, ctypes.byref(hdr), ctypes.sizeof(hdr))
                    self.buffers_played += 1
            time.sleep(0.004)

    def close(self) -> None:
        self._running = False
        self._thread.join()
        self._mm.waveOutReset(self._handle)
        for hdr in self._hdrs:
            self._mm.waveOutUnprepareHeader(self._handle, ctypes.byref(hdr), ctypes.sizeof(hdr))
        self._mm.waveOutClose(self._handle)


class MidiOut:  # pragma: no cover - needs a Windows MIDI device
    """The default MIDI output (Microsoft GS Wavetable Synth)."""

    def __init__(self) -> None:
        self._mm = _winmm()
        self._handle = ctypes.c_void_p()
        if self._mm.midiOutOpen(ctypes.byref(self._handle), ctypes.c_uint(_WAVE_MAPPER), 0, 0, 0):
            raise OSError("no MIDI output device (midiOutOpen failed)")
        self._lock = threading.Lock()

    def send(self, status: int, data1: int = 0, data2: int = 0) -> None:
        with self._lock:
            self._mm.midiOutShortMsg(self._handle, ctypes.c_uint(status | data1 << 8 | data2 << 16))

    def reset(self) -> None:
        with self._lock:
            self._mm.midiOutReset(self._handle)

    def close(self) -> None:
        self.reset()
        self._mm.midiOutClose(self._handle)
