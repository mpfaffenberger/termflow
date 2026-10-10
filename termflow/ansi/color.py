"""HSV/RGB color utilities for ANSI terminal colors.

This module provides color conversion functions and ANSI escape sequence
generators. Colors are emitted as 24-bit (truecolor) sequences when the
terminal supports them and as the nearest xterm 256-color index otherwise.

Terminals without truecolor support (notably macOS Terminal.app) don't ignore
``ESC[38;2;R;G;Bm``: they read each number as its own SGR code, so an RGB
component such as 104 or 106 becomes a bright blue or cyan *background*.
"""

import os
import re
import sys
from collections.abc import Mapping
from typing import Literal

# Regex for hex color validation
_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")

ColorMode = Literal["truecolor", "256"]
_COLOR_MODES: tuple[ColorMode, ...] = ("truecolor", "256")
_color_mode: ColorMode | None = None  # None: detect on first use

# Channel levels of the xterm 6x6x6 color cube (indices 16-231).
_CUBE_LEVELS = (0, 95, 135, 175, 215, 255)


def detect_color_mode(
    environ: Mapping[str, str] | None = None, platform: str | None = None
) -> ColorMode:
    """Detect whether the terminal supports truecolor, the same way Rich does.

    ``COLORTERM=truecolor`` (or ``24bit``) is the standard advertisement;
    modern Windows consoles support truecolor without setting it. Anything
    else gets the 256-color palette, which virtually every terminal supports.

    Args:
        environ: Environment to inspect (defaults to ``os.environ``).
        platform: Platform name to assume (defaults to ``sys.platform``).

    Returns:
        ``"truecolor"`` or ``"256"``.
    """
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if environ.get("COLORTERM", "").strip().lower() in ("truecolor", "24bit"):
        return "truecolor"
    if platform == "win32":
        return "truecolor"
    return "256"


def get_color_mode() -> ColorMode:
    """Return the active color mode, detecting it on first use."""
    global _color_mode
    if _color_mode is None:
        _color_mode = detect_color_mode()
    return _color_mode


def set_color_mode(mode: ColorMode | None) -> None:
    """Force a color mode, or pass ``None`` to go back to auto-detection.

    Applications that already know the terminal's capabilities can pass them
    on here so termflow agrees with the rest of their output.

    Raises:
        ValueError: If ``mode`` is not ``"truecolor"``, ``"256"`` or ``None``.
    """
    global _color_mode
    if mode is not None and mode not in _COLOR_MODES:
        raise ValueError(f"color mode must be one of {_COLOR_MODES} or None, not {mode!r}")
    _color_mode = mode


def rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """Return the closest xterm 256-color index (16-255) for an RGB color.

    Picks whichever of the 6x6x6 color cube and the 24-step greyscale ramp
    is nearer, the same approach tmux uses.

    Example:
        >>> rgb_to_ansi256(255, 0, 0)
        196
        >>> rgb_to_ansi256(128, 128, 128)
        244
    """

    def cube_index(value: int) -> int:
        if value < 48:
            return 0
        if value < 115:
            return 1
        return (value - 35) // 40

    def distance(first: tuple[int, int, int], second: tuple[int, int, int]) -> int:
        return sum((a - b) ** 2 for a, b in zip(first, second, strict=True))

    rgb = (r, g, b)
    red, green, blue = (cube_index(value) for value in rgb)
    cube_rgb = (_CUBE_LEVELS[red], _CUBE_LEVELS[green], _CUBE_LEVELS[blue])
    cube_color = 16 + 36 * red + 6 * green + blue

    average = sum(rgb) // 3
    grey_index = 23 if average > 238 else max(0, (average - 3) // 10)
    grey_level = 8 + 10 * grey_index
    grey_rgb = (grey_level, grey_level, grey_level)

    if distance(grey_rgb, rgb) < distance(cube_rgb, rgb):
        return 232 + grey_index
    return cube_color


def _color_sequence(hex_color: str, plane: int) -> str:
    """Build an SGR color sequence; ``plane`` is 38 (foreground) or 48 (background)."""
    rgb = hex2rgb(hex_color)
    if rgb is None:
        return ""
    if get_color_mode() == "256":
        return f"\x1b[{plane};5;{rgb_to_ansi256(*rgb)}m"
    r, g, b = rgb
    return f"\x1b[{plane};2;{r};{g};{b}m"


def hex2rgb(hex_color: str) -> tuple[int, int, int] | None:
    """Convert hex color (#RRGGBB or RRGGBB) to RGB tuple.

    Args:
        hex_color: Hex color string, with or without leading '#'.

    Returns:
        Tuple of (R, G, B) values (0-255 each), or None if invalid.

    Example:
        >>> hex2rgb("#FF5500")
        (255, 85, 0)
        >>> hex2rgb("00FF00")
        (0, 255, 0)
        >>> hex2rgb("invalid")
        None
    """
    match = _HEX_COLOR_RE.match(hex_color)
    if not match:
        return None

    hex_str = match.group(1)
    return (
        int(hex_str[0:2], 16),
        int(hex_str[2:4], 16),
        int(hex_str[4:6], 16),
    )


def rgb2hex(r: int, g: int, b: int) -> str:
    """Convert RGB values to hex string (#RRGGBB).

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).

    Returns:
        Hex color string with leading '#'.

    Example:
        >>> rgb2hex(255, 85, 0)
        '#FF5500'
        >>> rgb2hex(0, 0, 0)
        '#000000'
    """
    # Clamp values to valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV color to RGB.

    Args:
        h: Hue (0.0-1.0).
        s: Saturation (0.0-1.0).
        v: Value/brightness (0.0-1.0).

    Returns:
        Tuple of (R, G, B) values (0-255 each).

    Example:
        >>> hsv_to_rgb(0.0, 1.0, 1.0)  # Red
        (255, 0, 0)
        >>> hsv_to_rgb(0.333, 1.0, 1.0)  # Green-ish
        (0, 255, 2)
        >>> hsv_to_rgb(0.0, 0.0, 1.0)  # White
        (255, 255, 255)
    """
    if s == 0.0:
        # Achromatic (grey)
        val = int(v * 255)
        return (val, val, val)

    h = h % 1.0  # Wrap hue to 0-1 range
    h *= 6.0
    i = int(h)
    f = h - i

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return (int(r * 255), int(g * 255), int(b * 255))


def fg_color(hex_color: str) -> str:
    """Generate foreground ANSI escape sequence from hex color.

    Uses 24-bit truecolor ANSI escape codes (\x1b[38;2;R;G;Bm), or the
    nearest 256-color code (\x1b[38;5;Nm) when the color mode is ``"256"``.

    Args:
        hex_color: Hex color string (#RRGGBB or RRGGBB).

    Returns:
        ANSI escape sequence for foreground color, or empty string if invalid.

    Example:
        >>> fg_color("#FF5500")
        '\\x1b[38;2;255;85;0m'
        >>> fg_color("invalid")
        ''
    """
    return _color_sequence(hex_color, 38)


def bg_color(hex_color: str) -> str:
    """Generate background ANSI escape sequence from hex color.

    Uses 24-bit truecolor ANSI escape codes (\x1b[48;2;R;G;Bm), or the
    nearest 256-color code (\x1b[48;5;Nm) when the color mode is ``"256"``.

    Args:
        hex_color: Hex color string (#RRGGBB or RRGGBB).

    Returns:
        ANSI escape sequence for background color, or empty string if invalid.

    Example:
        >>> bg_color("#FF5500")
        '\\x1b[48;2;255;85;0m'
        >>> bg_color("invalid")
        ''
    """
    return _color_sequence(hex_color, 48)
