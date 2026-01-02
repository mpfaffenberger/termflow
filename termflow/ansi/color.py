"""HSV/RGB color utilities for ANSI terminal colors.

This module provides color conversion functions and ANSI escape sequence
generators for 24-bit (truecolor) terminal support.
"""

import re

# Regex for hex color validation
_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


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

    Uses 24-bit truecolor ANSI escape codes (\x1b[38;2;R;G;Bm).

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
    rgb = hex2rgb(hex_color)
    if rgb is None:
        return ""
    r, g, b = rgb
    return f"\x1b[38;2;{r};{g};{b}m"


def bg_color(hex_color: str) -> str:
    """Generate background ANSI escape sequence from hex color.

    Uses 24-bit truecolor ANSI escape codes (\x1b[48;2;R;G;Bm).

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
    rgb = hex2rgb(hex_color)
    if rgb is None:
        return ""
    r, g, b = rgb
    return f"\x1b[48;2;{r};{g};{b}m"
