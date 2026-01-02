"""HTML entity decoding utilities.

Provides functions to decode HTML entities in markdown content,
such as &amp; &lt; &gt; &copy; &#65; etc.
"""

import html
import re

# Pattern to match HTML entities
ENTITY_RE = re.compile(r"&(#?[a-zA-Z0-9]+);")


def decode_html_entities(text: str) -> str:
    """Decode HTML entities in text.

    Handles named entities (&amp;, &copy;), decimal (&#65;),
    and hexadecimal (&#x41;) character references.

    Args:
        text: Text potentially containing HTML entities.

    Returns:
        Text with entities decoded.

    Example:
        >>> decode_html_entities("Hello &amp; World")
        'Hello & World'
        >>> decode_html_entities("&copy; 2025")
        '© 2025'
        >>> decode_html_entities("&#65;&#66;&#67;")
        'ABC'
    """
    return html.unescape(text)


def has_html_entities(text: str) -> bool:
    """Check if text contains HTML entities.

    Args:
        text: Text to check.

    Returns:
        True if text contains at least one entity pattern.

    Example:
        >>> has_html_entities("Hello &amp; World")
        True
        >>> has_html_entities("Hello World")
        False
    """
    return bool(ENTITY_RE.search(text))


def decode_if_needed(text: str) -> str:
    """Decode HTML entities only if present (optimization).

    Args:
        text: Text potentially containing HTML entities.

    Returns:
        Decoded text, or original if no entities found.
    """
    if "&" in text and has_html_entities(text):
        return decode_html_entities(text)
    return text
