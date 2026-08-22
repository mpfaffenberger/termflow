"""Tests for termflow.themes (palettes + OSC engine)."""

import re
from io import StringIO

import pytest

from termflow.themes import (
    PALETTES,
    TerminalPalette,
    apply_palette,
    get_palette,
    palette_names,
    reset_palette,
    set_ansi_slot,
    set_bg,
    set_fg,
)
from termflow.themes.palette import blend_hex

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


# =============================================================================
# TerminalPalette + registry
# =============================================================================
class TestPaletteRegistry:
    def test_bundled_palettes_exist(self):
        for name in ("catppuccin_mocha", "tokyo_night", "purple_puppy", "solarized_light"):
            assert name in PALETTES

    def test_every_palette_is_well_formed(self):
        for name, palette in PALETTES.items():
            assert palette.name == name
            assert HEX_RE.match(palette.bg), f"{name}.bg"
            assert HEX_RE.match(palette.fg), f"{name}.fg"
            assert len(palette.ansi) == 16, f"{name}.ansi"
            for color in palette.ansi:
                assert HEX_RE.match(color), f"{name} slot {color}"

    def test_get_palette(self):
        assert get_palette("ocean") is PALETTES["ocean"]
        assert get_palette("nope") is None

    def test_get_palette_is_forgiving(self):
        assert get_palette("Tokyo-Night") is PALETTES["tokyo_night"]
        assert get_palette(" catppuccin mocha ") is PALETTES["catppuccin_mocha"]

    def test_palette_names_order(self):
        assert palette_names() == list(PALETTES)

    def test_wrong_slot_count_rejected(self):
        with pytest.raises(ValueError, match="16 ANSI colors"):
            TerminalPalette(name="bad", bg="#000000", fg="#ffffff", ansi=("#000000",) * 3)

    def test_dict_roundtrip(self):
        palette = PALETTES["forest"]
        assert TerminalPalette.from_dict("forest", palette.to_dict()) == palette

    def test_to_render_style_uses_bright_slots(self):
        palette = PALETTES["catppuccin_mocha"]
        style = palette.to_render_style()
        assert style.bright == palette.ansi[12]
        assert style.head == palette.ansi[10]
        assert style.error == palette.ansi[9]
        # Background tiers blend from bg toward fg.
        assert style.dark != palette.bg
        assert HEX_RE.match(style.dark)


class TestBlendHex:
    def test_endpoints(self):
        assert blend_hex("#000000", "#ffffff", 0.0) == "#000000"
        assert blend_hex("#000000", "#ffffff", 1.0) == "#FFFFFF"

    def test_midpoint(self):
        assert blend_hex("#000000", "#ffffff", 0.5) == "#808080"

    def test_clamps_t(self):
        assert blend_hex("#102030", "#ffffff", -5.0) == "#102030"

    def test_invalid_color_falls_back(self):
        assert blend_hex("nope", "#ffffff", 0.5) == "nope"


# =============================================================================
# OSC engine
# =============================================================================
class TestOscEngine:
    def test_set_bg_fg_slot_sequences(self):
        out = StringIO()
        set_bg("#112233", out)
        set_fg("#445566", out)
        set_ansi_slot(3, "#778899", out)
        value = out.getvalue()
        assert "\x1b]11;#112233\x07" in value
        assert "\x1b]10;#445566\x07" in value
        assert "\x1b]4;3;#778899\x07" in value

    def test_out_of_range_slot_ignored(self):
        out = StringIO()
        set_ansi_slot(16, "#ffffff", out)
        set_ansi_slot(-1, "#ffffff", out)
        assert out.getvalue() == ""

    def test_apply_palette_object(self):
        out = StringIO()
        palette = PALETTES["tokyo_night"]
        apply_palette(palette, out, register_reset=False)
        value = out.getvalue()
        assert f"\x1b]11;{palette.bg}\x07" in value
        assert f"\x1b]10;{palette.fg}\x07" in value
        for i, color in enumerate(palette.ansi):
            assert f"\x1b]4;{i};{color}\x07" in value

    def test_apply_partial_dict(self):
        out = StringIO()
        apply_palette({"bg": "#000000"}, out, register_reset=False)
        value = out.getvalue()
        assert "\x1b]11;#000000\x07" in value
        assert "\x1b]10;" not in value
        assert "\x1b]4;" not in value

    def test_apply_garbage_is_noop(self):
        out = StringIO()
        apply_palette("not a palette", out, register_reset=False)  # type: ignore[arg-type]
        assert out.getvalue() == ""

    def test_reset_palette_sequences(self):
        out = StringIO()
        reset_palette(out)
        value = out.getvalue()
        assert "\x1b]104\x07" in value
        assert "\x1b]110\x07" in value
        assert "\x1b]111\x07" in value

    def test_closed_stream_is_swallowed(self):
        out = StringIO()
        out.close()
        set_bg("#000000", out)  # must not raise
