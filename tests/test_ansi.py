"""Tests for termflow.ansi module."""

from termflow.ansi import (
    BOLD_OFF,
    BOLD_ON,
    RESET,
    bg_color,
    digit_to_superscript,
    fg_color,
    hex2rgb,
    hsv_to_rgb,
    is_ansi_code,
    number_to_superscript,
    rgb2hex,
    visible,
    visible_length,
    wrap_ansi,
)


class TestVisible:
    """Test ANSI stripping."""

    def test_strips_basic_codes(self):
        assert visible("\x1b[1mBold\x1b[0m") == "Bold"

    def test_strips_color_codes(self):
        assert visible("\x1b[38;2;255;0;0mRed\x1b[0m") == "Red"

    def test_strips_multiple_codes(self):
        text = "\x1b[1m\x1b[38;2;255;0;0mBold Red\x1b[0m\x1b[22m"
        assert visible(text) == "Bold Red"

    def test_no_codes(self):
        assert visible("Plain text") == "Plain text"

    def test_empty(self):
        assert visible("") == ""

    def test_only_codes(self):
        assert visible("\x1b[1m\x1b[0m") == ""


class TestVisibleLength:
    """Test visible length calculation."""

    def test_with_codes(self):
        assert visible_length("\x1b[1mHello\x1b[0m") == 5

    def test_plain(self):
        assert visible_length("Hello") == 5

    def test_cjk_double_width(self):
        # CJK characters are typically double-width
        length = visible_length("你好")
        assert length == 4  # 2 chars * 2 width each

    def test_empty(self):
        assert visible_length("") == 0

    def test_mixed_width(self):
        # Mix of ASCII and CJK
        length = visible_length("Hi你好")
        assert length == 6  # 2 ASCII + 4 CJK

    def test_emoji(self):
        # Emoji width varies, but should handle gracefully
        length = visible_length("Hello 👋")
        assert length >= 7


class TestColorConversion:
    """Test color conversion utilities."""

    def test_hex2rgb_with_hash(self):
        assert hex2rgb("#FF0000") == (255, 0, 0)

    def test_hex2rgb_without_hash(self):
        assert hex2rgb("00FF00") == (0, 255, 0)

    def test_hex2rgb_lowercase(self):
        assert hex2rgb("#aabbcc") == (170, 187, 204)

    def test_hex2rgb_blue(self):
        assert hex2rgb("#0000FF") == (0, 0, 255)

    def test_rgb2hex(self):
        # rgb2hex returns uppercase
        assert rgb2hex(255, 128, 64).lower() == "#ff8040"

    def test_rgb2hex_black(self):
        assert rgb2hex(0, 0, 0).lower() == "#000000"

    def test_rgb2hex_white(self):
        assert rgb2hex(255, 255, 255).lower() == "#ffffff"

    def test_hsv_to_rgb_red(self):
        r, g, b = hsv_to_rgb(0.0, 1.0, 1.0)
        assert r == 255 and g == 0 and b == 0

    def test_hsv_to_rgb_green(self):
        _r, g, _b = hsv_to_rgb(1 / 3, 1.0, 1.0)
        assert g == 255

    def test_hsv_to_rgb_blue(self):
        _r, _g, b = hsv_to_rgb(2 / 3, 1.0, 1.0)
        assert b == 255

    def test_hsv_to_rgb_white(self):
        r, g, b = hsv_to_rgb(0.0, 0.0, 1.0)
        assert r == 255 and g == 255 and b == 255

    def test_hsv_to_rgb_black(self):
        r, g, b = hsv_to_rgb(0.0, 0.0, 0.0)
        assert r == 0 and g == 0 and b == 0


class TestColorGeneration:
    """Test ANSI color code generation."""

    def test_fg_color(self):
        result = fg_color("#FF0000")
        assert "\x1b[38;2;255;0;0m" in result

    def test_bg_color(self):
        result = bg_color("#00FF00")
        assert "\x1b[48;2;0;255;0m" in result

    def test_fg_color_blue(self):
        result = fg_color("#0000FF")
        assert "\x1b[38;2;0;0;255m" in result


class TestAnsiDetection:
    """Test ANSI code detection."""

    def test_is_ansi_code_bold(self):
        assert is_ansi_code("\x1b[1m")

    def test_is_ansi_code_color(self):
        assert is_ansi_code("\x1b[38;2;255;0;0m")

    def test_is_ansi_code_reset(self):
        assert is_ansi_code("\x1b[0m")

    def test_is_ansi_code_false_plain(self):
        assert not is_ansi_code("hello")

    def test_is_ansi_code_false_empty(self):
        assert not is_ansi_code("")

    def test_is_ansi_code_false_partial(self):
        assert not is_ansi_code("\x1b[")


class TestSuperscript:
    """Test superscript conversion."""

    def test_digit_to_superscript_zero(self):
        assert digit_to_superscript(0) == "⁰"

    def test_digit_to_superscript_one(self):
        assert digit_to_superscript(1) == "¹"

    def test_digit_to_superscript_nine(self):
        assert digit_to_superscript(9) == "⁹"

    def test_number_to_superscript_single(self):
        assert number_to_superscript(5) == "⁵"

    def test_number_to_superscript_multi(self):
        assert number_to_superscript(42) == "⁴²"

    def test_number_to_superscript_large(self):
        assert number_to_superscript(123) == "¹²³"


class TestWrapAnsi:
    """Test ANSI-aware text wrapping."""

    def test_basic_wrap(self):
        lines = wrap_ansi("Hello world, this is a long line", 15)
        assert len(lines) >= 2

    def test_short_text_no_wrap(self):
        lines = wrap_ansi("Hello", 80)
        assert len(lines) == 1
        assert lines[0] == "Hello"

    def test_preserves_ansi(self):
        text = f"{BOLD_ON}Bold text here{BOLD_OFF}"
        lines = wrap_ansi(text, 20)
        # Should have ANSI codes in output
        combined = "".join(lines)
        assert "\x1b[" in combined

    def test_empty_string(self):
        lines = wrap_ansi("", 80)
        assert lines == [] or lines == [""]

    def test_exact_width(self):
        lines = wrap_ansi("12345", 5)
        assert len(lines) == 1


class TestConstants:
    """Test ANSI constants."""

    def test_reset_code(self):
        assert RESET == "\x1b[0m"

    def test_bold_on_code(self):
        assert BOLD_ON == "\x1b[1m"

    def test_bold_off_code(self):
        assert BOLD_OFF == "\x1b[22m"
