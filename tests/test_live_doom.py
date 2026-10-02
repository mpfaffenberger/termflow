"""Tests for the doom.wasm host, plus the engine features it needs.

The real engine test only runs when wasmtime is installed AND doom.wasm
is already cached -- the suite never touches the network.
"""

from __future__ import annotations

import hashlib
import time

import pytest

from termflow.live import (
    FramebufferView,
    KeyEvent,
    KeyState,
    LiveApp,
    Rect,
    ScreenBuffer,
    Widget,
    Window,
)
from termflow.live.demos import doom_wasm
from termflow.live.demos.doom_wasm import (
    DOOM_ASPECT,
    TAP_TICKS,
    DoomView,
    cache_dir,
    doom_key,
    doom_layout,
    fetch_doom_wasm,
)
from termflow.live.pixels import PixelSurface
from termflow.tui.keys import Key

CODES = {"KEY_UPARROW": 173, "KEY_FIRE": 163, "KEY_USE": 162, "KEY_ESCAPE": 27, "KEY_STRAFE_L": 160}


class FakeEngine:
    width, height = 4, 2
    finished = False

    def __init__(self):
        self.calls = []
        self.ticks = 0

    def key(self, name, pressed):
        self.calls.append((name, pressed))

    def tick(self):
        self.ticks += 1

    def frame(self):
        return [0xFF112233] * (self.width * self.height)  # alpha set: must be masked


class TestDoomKeys:
    def test_special_keys_use_exported_codes(self):
        assert doom_key(Key.UP, CODES) == 173
        assert doom_key("ctrl", CODES) == 163
        assert doom_key(" ", CODES) == 162
        assert doom_key(",", CODES) == 160

    def test_printable_ascii_is_its_own_code(self):
        assert doom_key("y", CODES) == ord("y")  # "quit? y" prompts, cheats
        assert doom_key("1", CODES) == ord("1")  # weapon slots

    def test_unknown_keys_are_ignored(self):
        assert doom_key("f5", CODES) is None
        assert doom_key("é", CODES) is None


class TestDoomView:
    def test_held_keys_become_down_then_up(self):
        engine = FakeEngine()
        view, keys = DoomView(engine), KeyState()
        keys.feed(KeyEvent(Key.UP), 0, True)
        view.update(0.03, keys)
        keys.end_frame()
        view.update(0.03, keys)  # still held: no repeat calls
        keys.feed(KeyEvent(Key.UP, pressed=False), 0, True)
        view.update(0.03, keys)
        assert engine.calls == [(Key.UP, True), (Key.UP, False)]
        assert engine.ticks == 3

    def test_tap_within_one_frame_stays_down_for_tap_ticks(self):
        engine = FakeEngine()
        view, keys = DoomView(engine), KeyState()
        keys.feed(KeyEvent(Key.ESCAPE), 0, True)
        keys.feed(KeyEvent(Key.ESCAPE, pressed=False), 0, True)
        for _ in range(TAP_TICKS):
            view.update(0.03, keys)
            keys.end_frame()
        assert engine.calls == [(Key.ESCAPE, True)]  # Doom samples key state
        view.update(0.03, keys)
        assert engine.calls == [(Key.ESCAPE, True), (Key.ESCAPE, False)]

    def test_holding_a_key_does_not_add_tap_ticks(self):
        engine = FakeEngine()
        view, keys = DoomView(engine), KeyState()
        keys.feed(KeyEvent("ctrl"), 0, True)
        view.update(0.03, keys)
        keys.end_frame()
        keys.feed(KeyEvent("ctrl", pressed=False), 0, True)
        view.update(0.03, keys)
        assert engine.calls == [("ctrl", True), ("ctrl", False)]

    def test_claims_tab_for_the_automap(self):
        view = DoomView(FakeEngine())
        assert view.on_key(KeyEvent(Key.TAB))
        assert not view.on_key(KeyEvent("x"))

    def test_frames_are_shown_masked_and_letterboxed(self):
        view = DoomView(FakeEngine())
        view.update(0.03, KeyState())
        buf = ScreenBuffer(8, 2)  # 8x4 px; 4:3 fits as 5x4, centered
        view.draw(buf.region(), focused=True)
        assert view.surface.pixels[1] == 0x112233
        assert view.surface.pixels[0] == 0  # letterbox bar

    def test_engine_exit_triggers_callback(self):
        engine = FakeEngine()
        exits = []
        view = DoomView(engine, on_exit=lambda: exits.append(1))
        engine.finished = True
        view.update(0.03, KeyState())
        assert exits == [1]


class TestFetch:
    def test_cached_file_with_good_hash_is_reused(self, tmp_path, monkeypatch):
        blob = b"pretend wasm"
        monkeypatch.setattr(doom_wasm, "DOOM_WASM_SHA256", hashlib.sha256(blob).hexdigest())
        (tmp_path / f"doom-{doom_wasm.DOOM_WASM_VERSION}.wasm").write_bytes(blob)

        def no_network(_url, _dest):
            raise AssertionError("should not download")

        assert fetch_doom_wasm(tmp_path, log=lambda _m: None, download=no_network).exists()

    def test_download_is_verified_and_cached(self, tmp_path, monkeypatch):
        blob = b"pretend wasm"
        monkeypatch.setattr(doom_wasm, "DOOM_WASM_SHA256", hashlib.sha256(blob).hexdigest())
        path = fetch_doom_wasm(
            tmp_path, log=lambda _m: None, download=lambda _u, dest: dest.write_bytes(blob)
        )
        assert path.read_bytes() == blob
        assert not list(tmp_path.glob("*.part"))

    def test_checksum_mismatch_refuses_to_run(self, tmp_path):
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            fetch_doom_wasm(
                tmp_path, log=lambda _m: None, download=lambda _u, dest: dest.write_bytes(b"evil")
            )
        assert not list(tmp_path.iterdir())

    def test_network_errors_become_runtime_errors(self, tmp_path):
        def offline(_url, _dest):
            raise OSError("no route to host")

        with pytest.raises(RuntimeError, match="could not download"):
            fetch_doom_wasm(tmp_path, log=lambda _m: None, download=offline)

    def test_cache_dir_respects_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setattr(doom_wasm.sys, "platform", "linux")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert cache_dir() == tmp_path / "termflow"


class TestLayout:
    def test_wide_terminal_gets_a_console_pane(self):
        doom, log = doom_layout(Rect(0, 0, 157, 41))
        assert doom.width == round(39 * 2 * DOOM_ASPECT) + 2
        assert doom.width + log.width == 157

    def test_narrow_terminal_is_all_doom(self):
        assert doom_layout(Rect(0, 0, 90, 41)) == [Rect(0, 0, 90, 41)]


class TestEngineFeatures:
    def test_blit_scaled_into_rect_masks_alpha(self):
        s = PixelSurface(4, 2, color=7)
        s.blit_scaled([0xAA000001, 0xBB000002], 2, 1, x=1, y=1, width=2, height=1)
        assert s.pixels == [7, 7, 7, 7, 7, 1, 2, 7]

    def test_framebuffer_aspect_letterboxes(self):
        view = FramebufferView(aspect=1.0, background=9)
        view.set_frame([5], 1, 1)
        view.draw(ScreenBuffer(4, 1).region(), focused=False)  # 4x2 px -> 2x2 centered
        assert view.surface.pixels == [9, 5, 5, 9, 9, 5, 5, 9]

    def test_key_state_exposes_held_and_pressed(self):
        ks = KeyState()
        ks.feed(KeyEvent("a"), 0, True)
        ks.feed(KeyEvent("b"), 0, True)
        ks.feed(KeyEvent("b", pressed=False), 0, True)
        assert ks.held == {"a"}
        assert ks.pressed == {"a", "b"}

    def test_focused_widget_can_claim_tab(self):
        class Greedy(Widget):
            def on_key(self, event):
                return event.key == Key.TAB

        app = LiveApp([Window("a", Greedy()), Window("b", Widget())], lambda a: [a, a])
        app.handle([KeyEvent(Key.TAB)], 0, True)
        assert app.focused is app.windows[0]
        assert app.keys.is_down(Key.TAB)  # still visible as held state

    def test_custom_status_hints(self):
        app = LiveApp(
            [Window("a", Widget())], lambda a: [a], hints="Esc menu", size=lambda: (40, 5)
        )
        assert "Esc menu" in app.step(0.03).row_text(4)
        assert "Tab focus" not in app.step(0.03).row_text(4)


def _cached_wasm():
    pytest.importorskip("wasmtime")
    path = cache_dir() / f"doom-{doom_wasm.DOOM_WASM_VERSION}.wasm"
    if not path.exists():
        pytest.skip("doom.wasm not cached (run `tf --doom-shareware` once)")
    return path


class TestRealDoom:
    def test_boots_renders_and_takes_input(self):
        engine = doom_wasm.DoomEngine(_cached_wasm())
        assert (engine.width, engine.height) == (640, 400)

        def tics(n):  # paced like the app, so every tick runs a 35 Hz tic
            for _ in range(n):
                engine.tick()
                time.sleep(0.03)

        tics(5)
        title = engine.frame()
        assert title is not None and max(title) <= 0xFFFFFF
        engine.key(Key.ESCAPE, True)
        tics(TAP_TICKS)
        engine.key(Key.ESCAPE, False)
        tics(5)
        assert engine.frame() != title  # the main menu opened
