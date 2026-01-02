"""Style pairs (on, off) tuples for easy toggling.

These tuples make it easy to wrap text with styles:
    styled_text = f"{BOLD[0]}bold text{BOLD[1]}"

Each tuple is (on_code, off_code) for symmetric style application.
"""

# =============================================================================
# Text Style Pairs: (on, off)
# =============================================================================

#: Bold text style - Note: BOLD_OFF (22m) also turns off DIM
BOLD: tuple[str, str] = ("\x1b[1m", "\x1b[22m")

#: Dim/faint text style - Note: DIM_OFF (22m) also turns off BOLD
DIM: tuple[str, str] = ("\x1b[2m", "\x1b[22m")

#: Italic text style
ITALIC: tuple[str, str] = ("\x1b[3m", "\x1b[23m")

#: Underline text style
UNDERLINE: tuple[str, str] = ("\x1b[4m", "\x1b[24m")

#: Strikethrough/strikeout text style
STRIKEOUT: tuple[str, str] = ("\x1b[9m", "\x1b[29m")

# =============================================================================
# OSC 8 Hyperlink
# =============================================================================

#: OSC 8 hyperlink - Usage: f"{LINK[0]}{url}\x1b\\{text}{LINK[1]}"
#: The URL goes after LINK[0], then ST (\x1b\\), then visible text, then LINK[1]
LINK: tuple[str, str] = ("\x1b]8;;", "\x1b]8;;\x1b\\")


def make_link(url: str, text: str) -> str:
    """Create an OSC 8 hyperlink.

    Args:
        url: The URL to link to.
        text: The visible text to display.

    Returns:
        ANSI-escaped hyperlink string.

    Example:
        >>> link = make_link("https://example.com", "Click here")
        >>> # Renders as clickable "Click here" in supported terminals
    """
    return f"{LINK[0]}{url}\x1b\\{text}{LINK[1]}"
