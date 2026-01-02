"""List rendering with bullet cycling.

Supports:
- Bullet (unordered) lists with cycling bullet characters
- Ordered (numbered) lists
- Nested lists with proper indentation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from termflow.ansi import RESET, fg_color

if TYPE_CHECKING:
    from termflow.render.style import RenderStyle

# =============================================================================
# Bullet Characters
# =============================================================================

# Bullets cycle by depth level for visual distinction
BULLETS = ["•", "◦", "▪", "▫", "▸", "▹"]

# Ordered list number styles by depth
ORDERED_STYLES = [
    lambda n: f"{n}.",  # 1. 2. 3.
    lambda n: f"{chr(ord('a') + (n - 1) % 26)})",  # a) b) c)
    lambda n: f"{to_roman(n).lower()}.",  # i. ii. iii.
    lambda n: f"{chr(ord('A') + (n - 1) % 26)})",  # A) B) C)
]


def to_roman(num: int) -> str:
    """Convert integer to Roman numeral."""
    if num <= 0 or num > 3999:
        return str(num)

    values = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]

    result = ""
    for value, numeral in values:
        while num >= value:
            result += numeral
            num -= value
    return result


def get_bullet(depth: int) -> str:
    """Get bullet character for nesting depth.

    Args:
        depth: Nesting level (0-based)

    Returns:
        Appropriate bullet character.

    Example:
        >>> get_bullet(0)
        '•'
        >>> get_bullet(1)
        '◦'
    """
    return BULLETS[depth % len(BULLETS)]


def get_ordered_bullet(number: int, depth: int) -> str:
    """Get formatted number for ordered list.

    Args:
        number: Item number (1-based)
        depth: Nesting level (0-based)

    Returns:
        Formatted number string.

    Example:
        >>> get_ordered_bullet(1, 0)
        '1.'
        >>> get_ordered_bullet(2, 1)
        'b)'
    """
    style_fn = ORDERED_STYLES[depth % len(ORDERED_STYLES)]
    return style_fn(number)


# Checkbox characters with better visibility
CHECKBOX_CHECKED = "[✓]"  # [✓]
CHECKBOX_UNCHECKED = "[ ]"


def render_list_item(
    depth: int,
    bullet_char: str,
    content: str,
    _width: int,  # Reserved for future text wrapping
    margin: str,
    style: RenderStyle,
    is_ordered: bool = False,
    number: int | None = None,
    checked: bool | None = None,
) -> list[str]:
    """Render a list item.

    Args:
        depth: Nesting depth (0-based)
        bullet_char: Bullet character or number
        content: Item content text
        width: Available width
        margin: Left margin string
        style: Render style
        is_ordered: Whether this is an ordered list
        number: Item number for ordered lists
        checked: For task lists: True=checked, False=unchecked, None=not a task

    Returns:
        Rendered lines (may be multiple if wrapped).
    """
    # Calculate indentation (2 spaces per depth level)
    indent = "  " * depth

    # Format bullet based on type
    if checked is not None:
        # Task list item - use checkbox with appropriate color
        if checked:
            # Green checkmark for completed
            green = fg_color("#50fa7b")  # Bright green
            formatted_bullet = f"{green}{CHECKBOX_CHECKED}{RESET}"
        else:
            # Grey for unchecked
            grey = fg_color(style.grey)
            formatted_bullet = f"{grey}{CHECKBOX_UNCHECKED}{RESET}"
    elif is_ordered and number is not None:
        fg = fg_color(style.symbol)
        bullet = get_ordered_bullet(number, depth)
        # Right-align numbers for consistent look
        bullet = bullet.rjust(3)
        formatted_bullet = f"{fg}{bullet}{RESET}"
    else:
        fg = fg_color(style.symbol)
        bullet = get_bullet(depth) if bullet_char == "•" else bullet_char
        formatted_bullet = f"{fg}{bullet}{RESET}"

    # Combine
    line = f"{margin}{indent}{formatted_bullet} {content}"

    return [line]


def render_list_continuation(
    depth: int,
    content: str,
    _width: int,  # Reserved for future text wrapping
    margin: str,
    _style: RenderStyle,  # Reserved for future styling
) -> list[str]:
    """Render continuation of a list item (wrapped content).

    Args:
        depth: Nesting depth
        content: Continuation text
        width: Available width
        margin: Left margin
        style: Render style

    Returns:
        Rendered lines.
    """
    # Indent to align with content after bullet
    indent = "  " * depth + "    "  # Extra 4 for bullet + space alignment

    return [f"{margin}{indent}{content}"]
