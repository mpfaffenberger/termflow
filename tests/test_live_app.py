"""Tests for the live frame loop, stock widgets, the demo, and Windows key mapping."""

from __future__ import annotations

import io
from itertools import pairwise

import pytest

from termflow.cli import create_parser
from termflow.live import (
    FramebufferView,
    KeyEvent,
    KeyState,
    LiveApp,
    LiveConsole,
    MarkdownView,
    PixelWidget,
    Rect,
    ScreenBuffer,
    TextLog,
    Widget,
    Window,
    hsplit,
    vsplit,
)
from termflow.live._win32 import translate_key
from termflow.live.console import ALT_SCREEN_OFF, ALT_SCREEN_ON, AUTOWRAP_ON
from termflow.live.demos.doom import Demon, RaycasterGame, build_app
from termflow.tui.keys import Key


class FakeIO:
    """Scripted input backend: one list of events per poll."""

    def __init__(self, script=(), reports_release=True):
        self.script = list(script)
        self.reports_release = reports_release
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def poll(self):
        return self.script.pop(0) if self.script else []


class Recorder(Widget):
    def __init__(self, focusable=True):
        self.focusable = focusable
        self.events = []
        self.held = []

    def on_key(self, event):
        self.events.append(event.key)
        return False  # observe only: consuming Tab would trap focus

    def update(self, _dt, keys):
        self.held.append(keys.is_down("w"))

    def draw(self, region, focused):
        region.text(0, 0, "focused" if focused else "idle")


def make_app(windows, layout=None, script=(), size=(40, 10), **kw):
    console = LiveConsole(output=io.StringIO(), io=FakeIO(script))
    ticks = iter(range(10_000))
    return LiveApp(
        windows,
        layout or (lambda area: hsplit(area, *([1] * len(windows)))),
        console=console,
        size=lambda: size,
        clock=lambda: next(ticks) / 30,
        sleep=lambda _s: None,
        **kw,
    )


class TestLayout:
    @pytest.mark.parametrize("total", [7, 10, 99])
    def test_splits_cover_area_without_gaps(self, total):
        rects = hsplit(Rect(3, 0, total, 5), 2, 1, 1)
        assert rects[0].x == 3
        assert sum(r.width for r in rects) == total
        for a, b in pairwise(rects):
            assert a.x + a.width == b.x

    def test_vsplit(self):
        top, bottom = vsplit(Rect(0, 0, 10, 9), 2, 1)
        assert (top.height, bottom.y, bottom.height) == (6, 6, 3)


class TestLiveApp:
    def test_step_draws_windows_and_status_bar(self):
        app = make_app([Window("One", Recorder()), Window("Two", Recorder())], title="demo")
        frame = app.step(0.03)
        assert frame.row_text(0).startswith("┏━ One ━")  # focused: heavy border
        assert "┌─ Two ─" in frame.row_text(0)
        assert "focused" in frame.row_text(1)
        assert "idle" in frame.row_text(1)
        assert frame.row_text(9).startswith(" demo")

    def test_tab_cycles_focus_skipping_unfocusable(self):
        a, b, c = Recorder(), Recorder(focusable=False), Recorder()
        app = make_app([Window("a", a), Window("b", b), Window("c", c)])
        app.handle([KeyEvent(Key.TAB)], 0, True)
        assert app.focused.widget is c
        app.handle([KeyEvent(Key.TAB)], 0, True)
        assert app.focused.widget is a

    def test_only_focused_widget_sees_keys(self):
        a, b = Recorder(), Recorder()
        app = make_app([Window("a", a), Window("b", b)])
        app.handle([KeyEvent("w"), KeyEvent("x")], 0, True)
        app.step(0.03)
        assert (a.held, b.held) == ([True], [False])
        assert (a.events, b.events) == (["w", "x"], [])

    def test_focus_change_releases_held_keys(self):
        a, b = Recorder(), Recorder()
        app = make_app([Window("a", a), Window("b", b)])
        app.handle([KeyEvent("w"), KeyEvent(Key.TAB)], 0, True)
        app.step(0.03)
        assert b.held == [False]

    def test_hidden_windows_cannot_keep_focus(self):
        a, b = Recorder(), Recorder()
        app = make_app([Window("a", a), Window("b", b)], layout=lambda area: [area])
        app.focus(app.windows[1])
        app.step(0.03)
        assert app.focused.widget is a

    def test_run_until_quit_and_restore_terminal(self):
        script = [[], [KeyEvent("ctrl-q")]]
        app = make_app([Window("a", Recorder())], script=script)
        stats = app.run(max_frames=50)
        out = app.console.output.getvalue()
        assert stats.frames == 2  # the quit lands during frame 2's input poll
        assert out.startswith(ALT_SCREEN_ON)
        assert out.endswith(AUTOWRAP_ON + "\x1b[?25h" + ALT_SCREEN_OFF)
        assert app.console._io.stopped

    def test_dt_is_clamped(self):
        seen = []

        class Probe(Widget):
            def update(self, dt, _keys):
                seen.append(dt)

        make_app([Window("p", Probe())]).step(5.0)
        assert seen == [0.1]


class TestWidgets:
    def test_text_log_is_file_like(self):
        log = TextLog()
        log.write("one\ntw")
        log.write("o\nthree")
        assert log.lines() == ["one", "two", "three"]
        buf = ScreenBuffer(10, 2)
        log.draw(buf.region(), focused=False)
        assert [buf.row_text(0).strip(), buf.row_text(1).strip()] == ["two", "three"]

    def test_markdown_view_streams_in(self):
        view = MarkdownView("# Title\n\nbody text", reveal_rate=4)
        buf = ScreenBuffer(30, 5)
        view.draw(buf.region(), focused=True)
        assert all(not buf.row_text(y).strip() for y in range(5))
        view.update(1.0, None)
        assert not view.done
        view.update(10.0, None)
        assert view.done
        view.draw(buf.region(), focused=True)
        assert "Title" in "".join(buf.row_text(y) for y in range(5))

    def test_markdown_view_scrolls_and_refollows(self):
        view = MarkdownView("\n\n".join(f"line {i}" for i in range(20)))
        region = ScreenBuffer(20, 4).region()
        view.draw(region, True)
        assert view.on_key(KeyEvent(Key.PAGE_UP))
        view.draw(region, True)
        assert not view._follow
        for _ in range(20):
            view.on_key(KeyEvent(Key.DOWN))
        view.draw(region, True)
        assert view._follow
        assert not view.on_key(KeyEvent("x"))

    def test_pixel_widget_surface_tracks_region(self):
        widget = PixelWidget()
        widget.draw(ScreenBuffer(8, 3).region(), focused=False)
        assert (widget.surface.width, widget.surface.height) == (8, 6)

    def test_framebuffer_view_scales_pushed_frames(self):
        view = FramebufferView()
        view.set_frame([0xFF0000, 0x00FF00], 2, 1)
        buf = ScreenBuffer(4, 1)
        view.draw(buf.region(), focused=False)
        assert buf.fg == [0xFF0000, 0xFF0000, 0x00FF00, 0x00FF00]


class TestWin32KeyTranslation:
    def test_letters_report_press_and_release(self):
        assert translate_key(0x57, True, "W", 0) == KeyEvent("w", text="W")
        assert translate_key(0x57, False, "w", 0) == KeyEvent("w", pressed=False, text="w")

    def test_ctrl_combo_and_altgr(self):
        assert translate_key(0x43, True, "\x03", 0x8).key == "ctrl-c"
        # AltGr (ctrl+alt) typing '@' on e.g. German layouts stays text.
        assert translate_key(0x51, True, "@", 0x8 | 0x1).key == "q"

    def test_named_and_modifier_keys(self):
        assert translate_key(0x26, True, "\x00", 0).key == Key.UP
        assert translate_key(0x10, False, "\x00", 0) == KeyEvent("shift", pressed=False)
        assert translate_key(0x20, True, " ", 0) == KeyEvent(" ", text=" ")

    def test_other_printables_and_unknowns(self):
        assert translate_key(0x31, True, "1", 0) == KeyEvent("1", text="1")
        assert translate_key(0xFF, True, "\x00", 0) is None


class TestDoomDemo:
    def test_app_renders_headless(self):
        app = build_app()
        app._size = lambda: (120, 30)
        frame = app.step(1 / 30)
        top = frame.row_text(0)
        assert "DOOM-ish" in top and "Briefing" in top
        assert "Kills 0/" in frame.row_text(29)

    def test_small_terminal_shows_only_the_game(self):
        app = build_app()
        app._size = lambda: (60, 20)
        assert "Briefing" not in app.step(1 / 30).row_text(0)

    def test_walking_forward_and_wall_collision(self):
        game = RaycasterGame()
        app = make_app([Window("g", game)])
        app.handle([KeyEvent("w")], 0, True)
        start = game.px
        for _ in range(100):
            app.step(0.1)
        assert game.px > start
        assert not game.solid(game.px, game.py)
        assert game.px < 7 - 0.2  # stopped by the wall at x=7

    def test_shooting_kills_a_demon(self):
        events = []
        game = RaycasterGame(on_event=events.append)
        game.demons = [Demon(game.px + 2, game.py)]
        for _ in range(2):
            game.fire_cooldown = 0
            game._fire()
        assert game.kills == 1
        assert game.cleared
        assert any("Level clear" in e for e in events)

    def test_walls_block_shots(self):
        game = RaycasterGame()
        game.demons = [Demon(game.px + 8, game.py)]  # behind the x=7 wall
        game._fire()
        assert game.demons[0].hp == 2

    def test_demons_hurt_and_reset_restores(self):
        game = RaycasterGame()
        game.demons = [Demon(game.px + 0.5, game.py)]
        game.update(0.05, KeyState())
        assert game.hp < 100
        game.on_key(KeyEvent("r"))  # only works when dead/cleared
        assert game.hp < 100
        game.hp = 0
        game.on_key(KeyEvent("r"))
        assert game.hp == 100

    def test_cli_has_doom_flag(self):
        assert create_parser().parse_args(["--doom"]).doom
