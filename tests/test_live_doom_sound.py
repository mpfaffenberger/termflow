"""Tests for doom.wasm sound: WAD/DMX/MUS parsing and the memory watcher.

Everything here is pure or uses a ctypes buffer as fake linear memory,
so it runs on any OS without audio devices.
"""

from __future__ import annotations

import array
import ctypes
import struct

import pytest

from termflow.live.demos import doom_sound as ds


def make_wad(lumps: dict[str, bytes], magic: bytes = b"IWAD") -> bytes:
    data = b"".join(lumps.values())
    directory = 12 + len(data)
    entries, pos = b"", 12
    for name, body in lumps.items():
        entries += struct.pack("<ii8s", pos, len(body), name.encode())
        pos += len(body)
    return magic + struct.pack("<ii", len(lumps), directory) + data + entries


def make_dmx(samples: bytes, rate: int = 11025) -> bytes:
    padded = b"\x80" * 16 + samples + b"\x80" * 16
    return struct.pack("<HHI", 3, rate, len(padded)) + padded


def make_mus(score: bytes) -> bytes:
    return b"MUS\x1a" + struct.pack("<HHHHHH", len(score), 16, 1, 0, 0, 0) + score


class TestWad:
    def test_read_wad(self):
        lumps = ds.read_wad(make_wad({"PLAYPAL": b"pal", "dspistol": b"bang"}))
        assert lumps == {"PLAYPAL": b"pal", "DSPISTOL": b"bang"}

    def test_rejects_non_wads(self):
        with pytest.raises(ValueError):
            ds.read_wad(b"nope" + b"\0" * 20)

    def test_finds_iwad_embedded_in_a_binary(self):
        blob = b"\0asm junk IWAD-but-not-really" + make_wad({"PLAYPAL": b"p", "DSOOF": b"o"})
        assert set(ds.find_embedded_iwad(blob)) == {"PLAYPAL", "DSOOF"}

    def test_user_wads_override_in_order(self, tmp_path):
        first, second = tmp_path / "a.wad", tmp_path / "b.wad"
        first.write_bytes(make_wad({"DSPISTOL": b"old", "D_E1M1": b"song"}))
        second.write_bytes(make_wad({"DSPISTOL": b"new"}, magic=b"PWAD"))
        lumps = ds.load_lumps(tmp_path / "unused.wasm", [first, second])
        assert lumps == {"DSPISTOL": b"new", "D_E1M1": b"song"}


class TestDmx:
    def test_decode_strips_padding_and_centers(self):
        rate, samples = ds.decode_dmx(make_dmx(bytes([128, 255, 0])))
        assert rate == 11025
        assert list(samples) == [0, 127 * ds.SFX_GAIN, -128 * ds.SFX_GAIN]

    def test_rejects_other_formats(self):
        with pytest.raises(ValueError):
            ds.decode_dmx(struct.pack("<HHI", 0, 11025, 0))

    def test_resample(self):
        clip = array.array("h", [1, 2, 3, 4])
        assert ds.resample(11025, clip, 11025) is clip
        assert list(ds.resample(22050, clip, 11025)) == [1, 3]


class TestMus:
    def test_events_channels_and_delays(self):
        score = bytes(
            [
                0x40, 0, 30,  # controller 0 (program 30) on channel 0
                0x91, 0x80 | 60, 100, 70,  # play note 60 @100 on ch 1, then delay 70
                0x01, 60,  # release on ch 1
                0x1F, 36,  # play note 36 on MUS channel 15 -> MIDI drums (9)
                0x2A, 128,  # pitch bend center on MUS ch 10 -> MIDI ch 11
                0xB0, 11, 0x81, 0x00,  # system: all notes off; delay 128 (two-byte VLQ)
                0x60,  # score end
            ]
        )  # fmt: skip
        events = ds.mus_to_midi(make_mus(score))
        assert events == [
            (0, 0xC0, 30, 0),
            (0, 0x91, 60, 100),
            (70, 0x81, 60, 0),
            (70, 0x99, 36, 127),  # drums default to velocity 127
            (70, 0xEB, 0, 64),  # 128 * 64 = 8192 = center
            (70, 0xB0, 123, 0),
        ]

    def test_velocity_is_remembered_per_channel(self):
        score = bytes([0x10, 0x80 | 60, 50, 0x10, 62, 0x60])
        notes = ds.mus_to_midi(make_mus(score))
        assert [e[3] for e in notes] == [50, 50]

    def test_rejects_non_mus(self):
        with pytest.raises(ValueError):
            ds.mus_to_midi(b"MThd....")


def fake_memory() -> tuple[ctypes.Array[ctypes.c_char], ds.SoundTables]:
    """Lay out S_music[68] + S_sfx[3] like doom.wasm does."""
    names_at = 64
    strings = b"e1m1\0intro\0"
    music_start = 256
    sfx_start = music_start + ds.MUSIC_COUNT * ds.MUSIC_STRIDE
    mem = bytearray(sfx_start + 3 * ds.SFX_STRIDE + 64)
    mem[names_at : names_at + len(strings)] = strings
    struct.pack_into("<i", mem, music_start + 16 * 1, names_at)  # e1m1
    struct.pack_into("<i", mem, music_start + 16 * 2, names_at + 5)  # intro
    for i, name in enumerate([b"none", b"pistol", b"radio"]):
        at = sfx_start + i * ds.SFX_STRIDE + ds.SFX_NAME_OFFSET
        mem[at : at + len(name) + 1] = name + b"\0"
    buf = ctypes.create_string_buffer(bytes(mem), len(mem))
    return buf, ds.SoundTables.locate(bytes(mem))


class TestSoundTables:
    def test_locate(self):
        _, tables = fake_memory()
        assert tables.sfx_names == ("none", "pistol", "radio")
        assert tables.music_names[1:3] == ("e1m1", "intro")

    def test_locate_fails_without_tables(self):
        with pytest.raises(ValueError):
            ds.SoundTables.locate(b"\0" * 1000)

    def test_current_song_follows_the_data_pointer(self):
        buf, tables = fake_memory()
        base = ctypes.addressof(buf)
        assert tables.current_song(base) is None
        struct.pack_into("<i", buf, tables.music_start + 16 * 2 + 8, 1234)  # intro.data
        assert tables.current_song(base) == "intro"

    def test_watcher_reports_each_start_once_per_tick(self):
        buf, tables = fake_memory()
        base = ctypes.addressof(buf)
        heard = []
        watcher = ds.SoundWatcher(tables, heard.append)
        pistol = tables.sfx_start + ds.SFX_STRIDE + ds.SFX_USEFULNESS_OFFSET

        watcher.begin_tick()
        struct.pack_into("<i", buf, pistol, 1)  # S_StartSound
        watcher.poll(base)
        watcher.poll(base)  # Doom asks for the time many times per tick
        struct.pack_into("<i", buf, pistol, 2)  # second channel, same tick
        watcher.poll(base)
        assert heard == ["pistol", "pistol"]

        struct.pack_into("<i", buf, pistol, 0)  # S_UpdateSounds stopped it
        watcher.begin_tick()
        struct.pack_into("<i", buf, pistol, 1)  # fired again next tick
        watcher.poll(base)
        assert heard == ["pistol", "pistol", "pistol"]

    def test_negative_usefulness_is_not_a_sound(self):
        buf, tables = fake_memory()
        heard = []
        watcher = ds.SoundWatcher(tables, heard.append)
        struct.pack_into("<i", buf, tables.sfx_start + ds.SFX_USEFULNESS_OFFSET, -1)  # never played
        watcher.poll(ctypes.addressof(buf))
        assert heard == []
