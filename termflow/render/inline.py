"""Inline markdown formatting: parsed inline tokens -> ANSI text.

Turns ``**bold**``, ``*italic*``, `` `code` ``, links, images and
footnotes into styled ANSI strings. The result is still a single
unwrapped string -- block renderers decide how to wrap it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from termflow.ansi import (
    BOLD_OFF,
    BOLD_ON,
    DIM_OFF,
    DIM_ON,
    ITALIC_OFF,
    ITALIC_ON,
    RESET,
    STRIKEOUT_OFF,
    STRIKEOUT_ON,
    UNDERLINE_OFF,
    UNDERLINE_ON,
    fg_color,
    make_link,
)
from termflow.parser.inline import InlineElement, InlineParser, InlineToken

if TYPE_CHECKING:
    from termflow.render.style import RenderFeatures, RenderStyle

#: (open, close) codes for inline elements that simply wrap their content.
_WRAPPERS: dict[InlineElement, tuple[str, str]] = {
    InlineElement.BOLD: (BOLD_ON, BOLD_OFF),
    InlineElement.ITALIC: (ITALIC_ON, ITALIC_OFF),
    InlineElement.BOLD_ITALIC: (f"{BOLD_ON}{ITALIC_ON}", f"{ITALIC_OFF}{BOLD_OFF}"),
    InlineElement.CODE: (DIM_ON, DIM_OFF),  # dim styling, no backticks
    InlineElement.UNDERLINE: (UNDERLINE_ON, UNDERLINE_OFF),
    InlineElement.STRIKEOUT: (STRIKEOUT_ON, STRIKEOUT_OFF),
}


def _format_link(token: InlineToken, style: RenderStyle, features: RenderFeatures) -> str:
    link_text = f"{fg_color(style.link)}{token.content}{RESET}"
    if features.hyperlinks and token.url:
        text = make_link(token.url, link_text)
    else:
        text = f"{UNDERLINE_ON}{link_text}{UNDERLINE_OFF}"
    return f"{text}{_url_suffix(token, style)}"


def _url_suffix(token: InlineToken, style: RenderStyle) -> str:
    """The grey `` (url)`` shown after links and images, if there is a URL."""
    return f" {fg_color(style.grey)}({token.url}){RESET}" if token.url else ""


def format_inline(text: str, style: RenderStyle, features: RenderFeatures) -> str:
    """Format ``text``'s inline markdown as an ANSI string.

    Args:
        text: Raw markdown text (a paragraph line, list item, cell...).
        style: Colors for links, images and footnotes.
        features: Feature flags (``hyperlinks`` enables OSC 8 links).

    Returns:
        The styled text, unwrapped.
    """
    parts: list[str] = []
    for token in InlineParser().parse(text):
        kind = token.element_type
        if kind in _WRAPPERS:
            on, off = _WRAPPERS[kind]
            parts.append(f"{on}{token.content}{off}")
        elif kind == InlineElement.LINK:
            parts.append(_format_link(token, style, features))
        elif kind == InlineElement.IMAGE:
            symbol_fg = fg_color(style.symbol)
            parts.append(f"{symbol_fg}[IMAGE: {token.content}]{RESET}{_url_suffix(token, style)}")
        elif kind == InlineElement.FOOTNOTE:
            parts.append(f"{fg_color(style.symbol)}[{token.content}]{RESET}")
        else:
            parts.append(token.content)
    return "".join(parts)
