"""Tests for termflow.tui (keys + menu builder)."""

from io import StringIO

from termflow.ansi.utils import visible
from termflow.tui import Key, MenuBuilder, MenuItem, parse_key


# =============================================================================
# Key parsing
# =============================================================================
class TestTerminalSession:
    """Refcounted raw-mode + alt-screen session for chained menus."""

    def test_single_session_enters_and_exits_once(self):
        from io import StringIO

        from termflow.tui import terminal_session

        out = StringIO()
        with terminal_session(out):
            pass
        assert out.getvalue().count("\x1b[?1049h") == 1
        assert out.getvalue().count("\x1b[?1049l") == 1

    def test_nested_sessions_share_one_alt_screen(self):
        from io import StringIO

        from termflow.tui import terminal_session

        out = StringIO()
        with terminal_session(out):
            with terminal_session(out):
                pass
            # Inner exit must NOT drop back to the primary screen.
            assert out.getvalue().count("\x1b[?1049l") == 0
        assert out.getvalue().count("\x1b[?1049h") == 1
        assert out.getvalue().count("\x1b[?1049l") == 1

    def test_session_releases_on_exception(self):
        from io import StringIO

        from termflow.tui import terminal_session

        out = StringIO()
        try:
            with terminal_session(out):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert out.getvalue().count("\x1b[?1049l") == 1
        # Depth back to zero: a fresh session works.
        with terminal_session(out):
            pass
        assert out.getvalue().count("\x1b[?1049h") == 2


class TestParseKey:
    def test_printable_passthrough(self):
        assert parse_key("a") == "a"
        assert parse_key("Z") == "Z"
        assert parse_key(" ") == " "
        assert parse_key("?") == "?"

    def test_enter_variants(self):
        assert parse_key("\r") == Key.ENTER
        assert parse_key("\n") == Key.ENTER

    def test_backspace_variants(self):
        assert parse_key("\x7f") == Key.BACKSPACE
        assert parse_key("\x08") == Key.BACKSPACE

    def test_tab(self):
        assert parse_key("\t") == Key.TAB

    def test_lone_escape(self):
        assert parse_key("\x1b") == Key.ESCAPE

    def test_arrow_keys(self):
        assert parse_key("\x1b", "[A") == Key.UP
        assert parse_key("\x1b", "[B") == Key.DOWN
        assert parse_key("\x1b", "[C") == Key.RIGHT
        assert parse_key("\x1b", "[D") == Key.LEFT

    def test_ss3_arrows(self):
        assert parse_key("\x1b", "OA") == Key.UP
        assert parse_key("\x1b", "OB") == Key.DOWN

    def test_paging_and_jump_keys(self):
        assert parse_key("\x1b", "[5~") == Key.PAGE_UP
        assert parse_key("\x1b", "[6~") == Key.PAGE_DOWN
        assert parse_key("\x1b", "[H") == Key.HOME
        assert parse_key("\x1b", "[F") == Key.END
        assert parse_key("\x1b", "[3~") == Key.DELETE

    def test_ctrl_keys(self):
        assert parse_key("\x03") == "ctrl-c"
        assert parse_key("\x01") == "ctrl-a"
        assert parse_key("\x1a") == "ctrl-z"

    def test_unknown_escape_returns_none(self):
        assert parse_key("\x1b", "[99z") is None


# =============================================================================
# Menu behavior (driven with scripted keys, no tty required)
# =============================================================================
class TestReadKey:
    """read_key against a real pipe -- the buffered-stream trap regression."""

    def _pipe_read_key(self, payload: bytes, keys: int = 1):
        import io
        import os

        from termflow.tui.keys import read_key

        r, w = os.pipe()
        os.write(w, payload)
        os.close(w)
        stream = io.TextIOWrapper(os.fdopen(r, "rb"), encoding="utf-8")
        try:
            return [read_key(stream) for _ in range(keys)]
        finally:
            stream.close()

    def test_arrow_burst_is_not_a_bare_escape(self):
        # ESC [ A arrives as one burst; TextIOWrapper.read(1) would slurp
        # it all into the internal buffer, fooling select() and turning
        # the arrow into a menu-cancelling ESC. The os.read path must win.
        assert self._pipe_read_key(b"\x1b[A") == [Key.UP]

    def test_sequence_of_arrows_then_enter(self):
        assert self._pipe_read_key(b"\x1b[B\x1b[B\r", keys=3) == [
            Key.DOWN,
            Key.DOWN,
            Key.ENTER,
        ]

    def test_lone_escape_still_cancels(self):
        assert self._pipe_read_key(b"\x1b") == [Key.ESCAPE]

    def test_utf8_multibyte_char(self):
        assert self._pipe_read_key("\u00e9".encode()) == ["\u00e9"]

    def test_stringio_fallback_still_works(self):
        from io import StringIO

        from termflow.tui.keys import read_key

        assert read_key(StringIO("a")) == "a"
        assert read_key(StringIO("")) == Key.ESCAPE


def run_menu(items, keys, **kwargs):
    """Drive a menu with a scripted key sequence, capturing output."""
    script = iter(keys)
    out = StringIO()
    builder = (
        MenuBuilder(kwargs.pop("title", "Test Menu"))
        .items(items)
        .output(out)
        .key_source(lambda: next(script))
        .size(lambda: (80, 24))
        .alt_screen(False)
    )
    for name, value in kwargs.items():
        getattr(builder, name)(value)
    return builder.run(), out.getvalue()


class TestMenuSingleSelect:
    def test_enter_selects_first_item(self):
        result, _ = run_menu(["alpha", "beta"], [Key.ENTER])
        assert not result.cancelled
        assert result.item.label == "alpha"
        assert result.item.value == "alpha"

    def test_down_then_enter(self):
        result, _ = run_menu(["alpha", "beta", "gamma"], [Key.DOWN, Key.ENTER])
        assert result.item.label == "beta"

    def test_up_wraps_to_bottom(self):
        result, _ = run_menu(["alpha", "beta", "gamma"], [Key.UP, Key.ENTER])
        assert result.item.label == "gamma"

    def test_escape_cancels(self):
        result, _ = run_menu(["alpha"], [Key.ESCAPE])
        assert result.cancelled
        assert result.item is None

    def test_ctrl_c_cancels(self):
        result, _ = run_menu(["alpha"], ["ctrl-c"])
        assert result.cancelled

    def test_custom_values_returned(self):
        items = [MenuItem("Pretty Label", value=42)]
        result, _ = run_menu(items, [Key.ENTER])
        assert result.item.value == 42

    def test_disabled_items_are_skipped(self):
        items = [
            MenuItem("-- section --", disabled=True),
            MenuItem("real"),
        ]
        result, _ = run_menu(items, [Key.ENTER])
        assert result.item.label == "real"

    def test_end_and_home(self):
        result, _ = run_menu(["a", "b", "c"], [Key.END, Key.ENTER])
        assert result.item.label == "c"
        result, _ = run_menu(["a", "b", "c"], [Key.END, Key.HOME, Key.ENTER])
        assert result.item.label == "a"


class TestMenuMultiSelect:
    def test_space_toggles_and_enter_returns_all(self):
        keys = [" ", Key.DOWN, " ", Key.ENTER]
        result, _ = run_menu(["a", "b", "c"], keys, multi_select=True)
        assert [it.label for it in result.items] == ["a", "b"]

    def test_toggle_off(self):
        keys = [" ", " ", Key.DOWN, " ", Key.ENTER]
        result, _ = run_menu(["a", "b"], keys, multi_select=True)
        assert [it.label for it in result.items] == ["b"]

    def test_enter_with_nothing_toggled_selects_highlighted(self):
        result, _ = run_menu(["a", "b"], [Key.DOWN, Key.ENTER], multi_select=True)
        assert [it.label for it in result.items] == ["b"]


class TestMenuSearch:
    def test_typing_filters_items(self):
        keys = ["g", "a", Key.ENTER]
        result, _ = run_menu(["alpha", "beta", "gamma"], keys, searchable=True)
        assert result.item.label == "gamma"

    def test_backspace_widens_filter(self):
        keys = ["z", "z", Key.BACKSPACE, Key.BACKSPACE, Key.ENTER]
        result, _ = run_menu(["alpha", "beta"], keys, searchable=True)
        assert result.item.label == "alpha"

    def test_enter_on_empty_filter_is_ignored(self):
        keys = ["z", Key.ENTER, Key.BACKSPACE, Key.ENTER]
        result, _ = run_menu(["alpha"], keys, searchable=True)
        assert not result.cancelled
        assert result.item.label == "alpha"

    def test_tall_preview_is_clamped_to_terminal_height(self):
        # A preview taller than the screen must not push the title and
        # list rows into scrollback (the frame would scroll and the menu
        # would appear as a blank left column next to a floating preview).
        tall = "\n".join(f"preview line {i}" for i in range(100))
        _result, screen = run_menu(
            [MenuItem("alpha"), MenuItem("beta")],
            [Key.ESCAPE],
            preview=lambda _item: tall,
            size=lambda: (100, 20),
        )
        # Painted frame fits in 20 rows.
        frame = screen.split("\x1b[H")[-1]
        painted_rows = frame.count("\r\n")
        assert painted_rows <= 20
        # Title and list rows survive; the preview tail is what gets cut.
        from termflow.ansi.utils import visible

        text = visible(screen)
        assert "alpha" in text
        assert "preview line 0" in text
        assert "preview line 99" not in text

    def test_custom_filter_fn(self):
        # Match against value, not label.
        result, _ = run_menu(
            [MenuItem("Pretty", value="ugly-internal"), MenuItem("Other", value="x")],
            ["u", "g", Key.ENTER],
            searchable=True,
            filter_fn=lambda q, it: q in str(it.value),
        )
        assert result.item.label == "Pretty"

    def test_no_matches_renders_hint(self):
        keys = ["z", Key.ESCAPE]
        _, screen = run_menu(["alpha"], keys, searchable=True)
        assert "(no matches)" in visible(screen)


class TestMenuRendering:
    def test_title_and_items_painted(self):
        _, screen = run_menu(["alpha", "beta"], [Key.ESCAPE], title="Pick One")
        text = visible(screen)
        assert "Pick One" in text
        assert "alpha" in text
        assert "beta" in text

    def test_descriptions_painted(self):
        items = [MenuItem("alpha", description="the first letter")]
        _, screen = run_menu(items, [Key.ESCAPE])
        assert "the first letter" in visible(screen)

    def test_pagination_hides_offpage_items(self):
        items = [f"item-{i:02d}" for i in range(25)]
        _, screen = run_menu(items, [Key.ESCAPE], page_size=10)
        text = visible(screen)
        assert "item-00" in text
        assert "item-15" not in text

    def test_page_down_reaches_next_page(self):
        items = [f"item-{i:02d}" for i in range(25)]
        result, _ = run_menu(items, [Key.PAGE_DOWN, Key.ENTER], page_size=10)
        assert result.item.label == "item-10"

    def test_preview_pane_renders(self):
        items = [MenuItem("alpha"), MenuItem("beta")]
        _, screen = run_menu(
            items,
            [Key.ESCAPE],
            preview=lambda it: f"details for {it.label}",
        )
        assert "details for alpha" in visible(screen)

    def test_on_highlight_fires_per_move(self):
        seen: list[str] = []
        run_menu(
            ["a", "b", "c"],
            [Key.DOWN, Key.DOWN, Key.ENTER],
            on_highlight=lambda it: seen.append(it.label),
        )
        # Initial highlight + two moves.
        assert seen == ["a", "b", "c"]

    def test_footer_hint_override(self):
        _, screen = run_menu(["a"], [Key.ESCAPE], footer_hint="custom help")
        text = visible(screen)
        assert "custom help" in text
        assert "esc cancel" not in text

    def test_initial_index_opens_on_item(self):
        result, _ = run_menu(["a", "b", "c"], [Key.ENTER], initial_index=2)
        assert result.item.label == "c"

    def test_list_width_controls_left_column(self):
        _, screen = run_menu(
            [MenuItem("item")],
            [Key.ESCAPE],
            preview=lambda _it: "PREVIEWTEXT",
            list_width=30,
        )
        line = next(ln for ln in visible(screen).splitlines() if "PREVIEWTEXT" in ln)
        assert line.index("PREVIEWTEXT") == 33  # 30 + " | " divider

    def test_on_key_handler_exits_with_result(self):
        from termflow.tui.menu import MenuResult

        def pin(_menu, item):
            return MenuResult(item=MenuItem(f"pinned:{item.label}"))

        script = iter([Key.DOWN, "p"])
        out = StringIO()
        result = (
            MenuBuilder("t")
            .items(["a", "b"])
            .output(out)
            .key_source(lambda: next(script))
            .size(lambda: (80, 24))
            .alt_screen(False)
            .on_key("p", pin)
            .run()
        )
        assert result.item.label == "pinned:b"

    def test_on_key_handler_can_mutate_and_continue(self):
        def reload(menu, _item):
            menu.replace_items([MenuItem("fresh")])
            return None

        script = iter(["r", Key.ENTER])
        out = StringIO()
        result = (
            MenuBuilder("t")
            .items(["stale"])
            .output(out)
            .key_source(lambda: next(script))
            .size(lambda: (80, 24))
            .alt_screen(False)
            .on_key("r", reload)
            .run()
        )
        assert result.item.label == "fresh"

    def test_on_key_takes_precedence_over_search(self):
        from termflow.tui.menu import MenuResult

        script = iter(["p"])
        out = StringIO()
        result = (
            MenuBuilder("t")
            .items(["alpha"])
            .searchable()
            .output(out)
            .key_source(lambda: next(script))
            .size(lambda: (80, 24))
            .alt_screen(False)
            .on_key("p", lambda _menu, item: MenuResult(item=item))
            .run()
        )
        # "p" acted as the bound key, not a search character.
        assert result.item.label == "alpha"

    def test_preview_errors_swallowed(self):
        def boom(_item):
            raise RuntimeError("nope")

        result, _ = run_menu(["a"], [Key.ENTER], preview=boom)
        assert result.item.label == "a"
