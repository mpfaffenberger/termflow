"""Tests for termflow.tui (keys + menu builder)."""

from io import StringIO

from termflow.ansi.utils import visible
from termflow.tui import Key, MenuBuilder, MenuItem, parse_key


# =============================================================================
# Key parsing
# =============================================================================
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
