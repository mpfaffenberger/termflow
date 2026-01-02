"""ANSI module - terminal escape codes and styling.

This module provides comprehensive ANSI escape code support for terminal
rendering, including:

- Raw escape code constants (codes.py)
- Color conversion utilities (color.py)
- Style pairs for easy toggling (style.py)
- Text processing utilities (utils.py)

Example:
    >>> from termflow.ansi import BOLD, fg_color, visible_length
    >>> styled = f"{BOLD[0]}{fg_color('#FF5500')}Orange Bold{BOLD[1]}"
    >>> print(visible_length(styled))  # Returns 11, not counting ANSI codes
"""

from termflow.ansi.codes import (
    BGRESET,
    BOLD_OFF,
    BOLD_ON,
    DIM_OFF,
    DIM_ON,
    ITALIC_OFF,
    ITALIC_ON,
    RESET,
    STRIKEOUT_OFF,
    STRIKEOUT_ON,
    SUPERSCRIPTS,
    UNDERLINE_OFF,
    UNDERLINE_ON,
    digit_to_superscript,
    number_to_superscript,
)
from termflow.ansi.color import (
    bg_color,
    fg_color,
    hex2rgb,
    hsv_to_rgb,
    rgb2hex,
)
from termflow.ansi.style import (
    BOLD,
    DIM,
    ITALIC,
    LINK,
    STRIKEOUT,
    UNDERLINE,
    make_link,
)
from termflow.ansi.utils import (
    ANSI_CSI_RE,
    ANSI_ESCAPE_RE,
    ANSI_SGR_RE,
    extract_ansi_codes,
    is_ansi_code,
    parse_sgr_params,
    split_ansi,
    truncate_ansi,
    visible,
    visible_length,
    wrap_ansi,
)

__all__ = [
    "ANSI_CSI_RE",
    "ANSI_ESCAPE_RE",
    "ANSI_SGR_RE",
    "BGRESET",
    "BOLD",
    "BOLD_OFF",
    "BOLD_ON",
    "DIM",
    "DIM_OFF",
    "DIM_ON",
    "ITALIC",
    "ITALIC_OFF",
    "ITALIC_ON",
    "LINK",
    "RESET",
    "STRIKEOUT",
    "STRIKEOUT_OFF",
    "STRIKEOUT_ON",
    "SUPERSCRIPTS",
    "UNDERLINE",
    "UNDERLINE_OFF",
    "UNDERLINE_ON",
    "bg_color",
    "digit_to_superscript",
    "extract_ansi_codes",
    "fg_color",
    "hex2rgb",
    "hsv_to_rgb",
    "is_ansi_code",
    "make_link",
    "number_to_superscript",
    "parse_sgr_params",
    "rgb2hex",
    "split_ansi",
    "truncate_ansi",
    "visible",
    "visible_length",
    "wrap_ansi",
]
