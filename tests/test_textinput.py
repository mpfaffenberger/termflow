"""Tests for termflow.tui.textinput -- headless scripted drives."""

from __future__ import annotations

from io import StringIO

from termflow.ansi.utils import visible_length
from termflow.render.style import RenderStyle
from termflow.tui import TextInputBuilder, TextInputResult


def drive(keys, **overrides):
    """Run a builder-configured input against a scripted key sequence."""
    script = iter(keys)
    output = StringIO()
    builder = (
        TextInputBuilder(overrides.pop("title", "Test Input"))
        .key_source(lambda: next(script))
        .output(output)
        .size(overrides.pop("size", lambda: (60, 12)))
        .alt_screen(False)
    )
    for name, value in overrides.items():
        method = getattr(builder, name)
        if name == "on_key":
            method(*value)
        else:
            method(value)
    widget = builder.build()
    result = widget.run()
    return widget, result, output.getvalue()


def frame_lines(raw):
    for frame in raw.split("\x1b[H")[1:]:
        for line in frame.replace("\x1b[J", "").split("\r\n"):
            yield line.replace("\x1b[K", "")


class TestBasicEditing:
    def test_type_and_commit(self):
        _, result, _ = drive(["h", "i", "enter"])
        assert result == TextInputResult(value="hi")

    def test_escape_cancels(self):
        _, result, _ = drive(["n", "o", "escape"])
        assert result.cancelled and result.value is None

    def test_ctrl_c_cancels(self):
        _, result, _ = drive(["ctrl-c"])
        assert result.cancelled

    def test_initial_value_edits_at_end(self):
        _, result, _ = drive(["!", "enter"], initial="hello")
        assert result.value == "hello!"

    def test_cursor_movement_and_insert(self):
        # "bc" -> home -> insert "a" -> end -> insert "d"
        _, result, _ = drive(["b", "c", "home", "a", "end", "d", "enter"])
        assert result.value == "abcd"

    def test_left_right_arrows(self):
        _, result, _ = drive(["a", "c", "left", "b", "right", "d", "enter"])
        assert result.value == "abcd"

    def test_backspace_and_delete(self):
        # "abcd" -> backspace kills d -> home -> delete kills a
        _, result, _ = drive(["a", "b", "c", "d", "backspace", "home", "delete", "enter"])
        assert result.value == "bc"

    def test_emacs_bindings(self):
        # ctrl-a home, ctrl-e end, ctrl-b/f movement
        _, result, _ = drive(["b", "c", "ctrl-a", "a", "ctrl-e", "d", "ctrl-b", "x", "enter"])
        assert result.value == "abcxd"


class TestKillCommands:
    def test_ctrl_u_kills_to_start(self):
        _, result, _ = drive(["a", "b", "c", "left", "ctrl-u", "enter"])
        assert result.value == "c"

    def test_ctrl_k_kills_to_end(self):
        _, result, _ = drive(["a", "b", "c", "home", "right", "ctrl-k", "enter"])
        assert result.value == "a"

    def test_ctrl_w_deletes_word_back(self):
        keys = [*"path/to/file", "ctrl-w", "enter"]
        _, result, _ = drive(keys)
        assert result.value == "path/to/"

    def test_ctrl_w_eats_trailing_separators(self):
        keys = [*"a b  ", "ctrl-w", "enter"]
        _, result, _ = drive(keys)
        assert result.value == "a "


class TestValidation:
    def test_validator_blocks_commit_and_shows_error(self):
        _, result, raw = drive(
            ["enter", "o", "k", "enter"],
            validator=lambda text: None if text else "value required",
        )
        assert result.value == "ok"
        assert "value required" in raw

    def test_error_clears_on_edit(self):
        widget, result, _ = drive(
            ["enter", "x", "backspace", "y", "enter"],
            validator=lambda text: None if text else "nope",
        )
        assert result.value == "y"
        assert widget._error is None


class TestRendering:
    def test_placeholder_shown_when_empty(self):
        _, _, raw = drive(["escape"], placeholder="sk-...")
        assert "sk-..." in raw

    def test_placeholder_gone_once_typed(self):
        _, _, raw = drive(["x", "escape"], placeholder="ghost")
        final_frame = raw.split("\x1b[H")[-1]
        assert "ghost" not in final_frame

    def test_mask_hides_value(self):
        _, result, raw = drive([*"secret", "enter"], mask="*")
        assert result.value == "secret"
        assert "secret" not in raw
        assert "******" in raw.replace("\x1b[K", "")

    def test_no_line_exceeds_terminal_width(self):
        keys = [
            *"a long value that will definitely overflow",
            "home",
            "end",
            "enter",
        ]
        for width in (20, 40, 80):
            _, _, raw = drive(keys, size=lambda w=width: (w, 10))
            for line in frame_lines(raw):
                assert visible_length(line) <= width, f"line overflows {width} cols: {line!r}"

    def test_long_value_scrolls_to_keep_cursor_visible(self):
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        _widget, result, raw = drive([*text, "enter"], size=lambda: (20, 10))
        assert result.value == text
        # The tail of the value (where the cursor lives) must be visible
        # in the final frame; the head scrolled away.
        final = raw.split("\x1b[H")[-1].replace("\x1b[K", "")
        assert "89" in final
        assert "abc" not in final

    def test_wide_characters_do_not_break_layout(self):
        keys = [*"日本語テキスト", "home", "right", "enter"]
        _, result, raw = drive(keys, size=lambda: (16, 10))
        assert result.value == "日本語テキスト"
        for line in frame_lines(raw):
            assert visible_length(line) <= 16

    def test_title_and_hint_render(self):
        _, _, raw = drive(["escape"], title="Speak, human")
        assert "Speak, human" in raw
        assert "Enter accept" in raw

    def test_custom_style_reaches_frame(self):
        from termflow.ansi.color import fg_color

        style = RenderStyle(bright="#123456")
        _, _, raw = drive(["escape"], style=style)
        assert fg_color("#123456") in raw


class TestFormComposition:
    def test_on_key_handler_ends_run_with_key(self):
        def focus_down(widget):
            return TextInputResult(value=widget.text, key="down")

        _, result, _ = drive(["a", "b", "down"], on_key=("down", focus_down))
        assert result == TextInputResult(value="ab", key="down")

    def test_on_key_handler_may_decline(self):
        def noop(_widget):
            return None

        _, result, _ = drive(["tab", "x", "enter"], on_key=("tab", noop))
        assert result.value == "x"

    def test_set_text_repositions_cursor(self):
        def reset(widget):
            widget.set_text("fresh")
            return None

        _widget, result, _ = drive(["o", "l", "d", "tab", "!", "enter"], on_key=("tab", reset))
        assert result.value == "fresh!"


def test_builder_on_key_signature_helper():
    """drive() passes on_key as a (key, handler) tuple -- verify direct use."""
    script = iter(["x", "enter"])
    result = (
        TextInputBuilder("direct")
        .on_key("ctrl-t", lambda _w: None)
        .key_source(lambda: next(script))
        .output(StringIO())
        .size(lambda: (40, 8))
        .alt_screen(False)
        .run()
    )
    assert result.value == "x"
