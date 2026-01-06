"""Terminal graphics output for Mermaid diagrams.

Converts PIL Images to various terminal graphics formats:
- Kitty Graphics Protocol (for Kitty terminal)
- iTerm2 Inline Images (for iTerm2)
- Sixel (detected but falls back to blocks)
- Unicode Block Characters (universal fallback)

Example:
    >>> from PIL import Image
    >>> from termflow.render.mermaid.graphics import image_to_terminal
    >>>
    >>> img = Image.new("RGB", (100, 50), "white")
    >>> print(image_to_terminal(img))
"""

from __future__ import annotations

import base64
import io
import os
from enum import Enum
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants
# =============================================================================

# Block characters for rendering
UPPER_HALF_BLOCK = "\u2580"  # ▀
LOWER_HALF_BLOCK = "\u2584"  # ▄
FULL_BLOCK = "\u2588"  # █

# ANSI escape sequences
RESET = "\x1b[0m"


# =============================================================================
# Enums
# =============================================================================


class GraphicsProtocol(Enum):
    """Available terminal graphics protocols."""

    KITTY = "kitty"  # Kitty Graphics Protocol
    ITERM2 = "iterm2"  # iTerm2 Inline Images
    SIXEL = "sixel"  # Sixel graphics
    BLOCK = "block"  # Unicode block characters (universal)
    NONE = "none"  # No graphics support


# =============================================================================
# Protocol Detection
# =============================================================================


def detect_graphics_protocol() -> GraphicsProtocol:
    """Detect the best available graphics protocol for the current terminal.

    Checks environment variables and terminal capabilities to determine
    which graphics protocol to use.

    Returns:
        The best available GraphicsProtocol
    """
    # Check for Kitty terminal
    if os.environ.get("KITTY_WINDOW_ID"):
        return GraphicsProtocol.KITTY

    # Check for iTerm2
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program == "iTerm.app":
        return GraphicsProtocol.ITERM2

    # Check for WezTerm (supports iTerm2 protocol)
    if term_program == "WezTerm":
        return GraphicsProtocol.ITERM2

    # Check for Sixel support (basic detection)
    # Many terminals advertise sixel in TERM or via DA1 response
    term = os.environ.get("TERM", "")
    if "sixel" in term.lower():
        # For now, fall back to blocks since Sixel encoding is complex
        # TODO: Implement proper Sixel encoding
        return GraphicsProtocol.BLOCK

    # Check for terminals known to support Sixel
    sixel_terminals = ["mlterm", "xterm-256color"]  # xterm with sixel
    if any(t in term.lower() for t in sixel_terminals):
        # Still fall back to blocks for now
        return GraphicsProtocol.BLOCK

    # Default to block characters (works everywhere)
    return GraphicsProtocol.BLOCK


def is_graphics_supported() -> bool:
    """Check if any graphics protocol is supported.

    Returns:
        True if at least block characters can be rendered
    """
    return detect_graphics_protocol() != GraphicsProtocol.NONE


# =============================================================================
# Kitty Graphics Protocol
# =============================================================================


def image_to_kitty(img: Image.Image) -> str:
    """Encode image for Kitty Graphics Protocol.

    The Kitty protocol uses:
    ESC_Ga=T,f=100,s=<width>,v=<height>;<base64 PNG>ESC\\

    Args:
        img: PIL Image to encode

    Returns:
        Escape sequence string for Kitty
    """
    # Ensure RGBA mode
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Encode as PNG
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    png_data = buffer.getvalue()

    # Base64 encode
    b64_data = base64.b64encode(png_data).decode("ascii")

    # Build Kitty escape sequence
    # a=T means transmit and display
    # f=100 means PNG format
    # s=width, v=height
    width, height = img.size

    # Kitty requires chunked transmission for large images
    # Each chunk can be up to 4096 bytes
    chunk_size = 4096
    chunks = [b64_data[i : i + chunk_size] for i in range(0, len(b64_data), chunk_size)]

    if len(chunks) == 1:
        # Single chunk - simple case
        return f"\x1b_Ga=T,f=100,s={width},v={height};{b64_data}\x1b\\"

    # Multiple chunks
    result = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk
            result.append(f"\x1b_Ga=T,f=100,s={width},v={height},m=1;{chunk}\x1b\\")
        elif i == len(chunks) - 1:
            # Last chunk
            result.append(f"\x1b_Gm=0;{chunk}\x1b\\")
        else:
            # Middle chunk
            result.append(f"\x1b_Gm=1;{chunk}\x1b\\")

    return "".join(result)


# =============================================================================
# iTerm2 Inline Images
# =============================================================================


def image_to_iterm2(img: Image.Image) -> str:
    """Encode image for iTerm2 Inline Images protocol.

    The iTerm2 protocol uses:
    ESC]1337;File=inline=1;width=auto;height=auto:<base64>BEL

    Args:
        img: PIL Image to encode

    Returns:
        Escape sequence string for iTerm2
    """
    # Ensure RGBA mode
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Encode as PNG
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    png_data = buffer.getvalue()

    # Base64 encode
    b64_data = base64.b64encode(png_data).decode("ascii")

    # Build iTerm2 escape sequence
    # inline=1 means display inline
    # preserveAspectRatio=1 maintains proportions
    return f"\x1b]1337;File=inline=1;preserveAspectRatio=1:{b64_data}\x07"


# =============================================================================
# Sixel Graphics
# =============================================================================


def image_to_sixel(img: Image.Image) -> str:
    """Encode image as Sixel graphics.

    Sixel is a complex format. This implementation falls back to block
    characters for now. A proper implementation would:
    1. Quantize image to 256 colors
    2. Build color palette
    3. Encode pixels as sixel data (6 rows at a time)

    Args:
        img: PIL Image to encode

    Returns:
        Block character representation (fallback)
    """
    # TODO: Implement proper Sixel encoding
    # For now, fall back to block characters
    return image_to_blocks(img)


# =============================================================================
# Block Character Rendering
# =============================================================================


def _rgb_to_ansi_fg(r: int, g: int, b: int) -> str:
    """Generate ANSI 24-bit foreground color escape sequence."""
    return f"\x1b[38;2;{r};{g};{b}m"


def _rgb_to_ansi_bg(r: int, g: int, b: int) -> str:
    """Generate ANSI 24-bit background color escape sequence."""
    return f"\x1b[48;2;{r};{g};{b}m"


def _colors_similar(c1: tuple[int, ...], c2: tuple[int, ...], threshold: int = 10) -> bool:
    """Check if two colors are similar within a threshold."""
    if len(c1) < 3 or len(c2) < 3:
        return False
    return all(abs(c1[i] - c2[i]) <= threshold for i in range(3))


def image_to_blocks(
    img: Image.Image,
    width: int | None = None,
    maintain_aspect: bool = True,
) -> str:
    """Convert image to Unicode block characters with 24-bit color.

    Uses half-block characters (▀ ▄) to achieve 2x vertical resolution.
    Each character cell represents 2 vertical pixels.

    Args:
        img: PIL Image to convert
        width: Target width in characters (None = use image width / 2)
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        String with ANSI color codes and block characters
    """
    # Convert to RGB if necessary
    if img.mode == "RGBA":
        # Composite onto white background for transparency
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha as mask
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    orig_width, orig_height = img.size

    # Calculate target dimensions
    if width is None:
        # Default: 1 character = 2 pixels width
        target_width = orig_width // 2
    else:
        target_width = width

    # Ensure minimum size
    target_width = max(target_width, 10)

    if maintain_aspect:
        # Calculate height maintaining aspect ratio
        # Each char is roughly 2:1 (2 pixels wide, 1 pixel tall visually)
        # But we use half-blocks so each char = 2 vertical pixels
        aspect_ratio = orig_height / orig_width
        # Terminal chars are ~2:1 aspect, and we use half-blocks (2 pixels/char)
        target_height = int(target_width * aspect_ratio)
    else:
        target_height = orig_height // 2

    # Ensure even height (we process 2 rows at a time)
    target_height = max(target_height, 2)
    if target_height % 2 != 0:
        target_height += 1

    # Resize image
    # Width = target_width (1 char per pixel after resize)
    # Height = target_height * 2 (2 pixels per char row, using half-blocks)
    new_width = target_width
    new_height = target_height * 2

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Get pixel data
    pixels = img.load()

    # Build output
    lines = []

    for y in range(0, new_height, 2):
        line_parts = []
        prev_fg = None
        prev_bg = None

        for x in range(new_width):
            # Get top and bottom pixel colors
            top_color = pixels[x, y]
            bottom_color = pixels[x, y + 1] if y + 1 < new_height else top_color

            # Determine which character and colors to use
            if _colors_similar(top_color, bottom_color):
                # Both pixels same color - use full block or space
                fg = top_color
                bg = top_color
                char = FULL_BLOCK
            else:
                # Different colors - use upper half block
                # Upper half block: foreground = top, background = bottom
                fg = top_color
                bg = bottom_color
                char = UPPER_HALF_BLOCK

            # Build color codes (only emit if changed)
            color_codes = ""
            if fg != prev_fg:
                color_codes += _rgb_to_ansi_fg(fg[0], fg[1], fg[2])
                prev_fg = fg
            if bg != prev_bg:
                color_codes += _rgb_to_ansi_bg(bg[0], bg[1], bg[2])
                prev_bg = bg

            line_parts.append(color_codes + char)

        # Reset at end of line
        lines.append("".join(line_parts) + RESET)

    return "\n".join(lines)


# =============================================================================
# Main Interface
# =============================================================================


def image_to_terminal(
    img: Image.Image,
    width: int | None = None,
    protocol: GraphicsProtocol | None = None,
) -> str:
    """Convert image to terminal escape sequences.

    Automatically detects the best graphics protocol if not specified.

    Args:
        img: PIL Image to convert
        width: Target width in characters (for block mode)
        protocol: Graphics protocol to use (auto-detect if None)

    Returns:
        String with escape sequences for terminal display
    """
    if protocol is None:
        protocol = detect_graphics_protocol()

    match protocol:
        case GraphicsProtocol.KITTY:
            return image_to_kitty(img)
        case GraphicsProtocol.ITERM2:
            return image_to_iterm2(img)
        case GraphicsProtocol.SIXEL:
            return image_to_sixel(img)
        case GraphicsProtocol.BLOCK:
            return image_to_blocks(img, width)
        case GraphicsProtocol.NONE:
            return "[Image display not supported]\n"
        case _:
            return image_to_blocks(img, width)


def render_mermaid_to_terminal(
    mermaid_code: str,
    width: int | None = None,
    protocol: GraphicsProtocol | None = None,
    render_config: "RenderConfig | None" = None,
) -> str:
    """Convenience function to render Mermaid code directly to terminal.

    Parses, lays out, renders to image, and converts to terminal graphics.

    Args:
        mermaid_code: Mermaid diagram source code
        width: Target width in characters
        protocol: Graphics protocol (auto-detect if None)
        render_config: Optional RenderConfig for customizing diagram appearance

    Returns:
        Terminal escape sequences for display

    Raises:
        MermaidParseError: If parsing fails
    """
    from termflow.render.mermaid.canvas import RenderConfig, render_to_image
    from termflow.render.mermaid.layout import layout_graph
    from termflow.render.mermaid.parser import parse_mermaid

    graph = parse_mermaid(mermaid_code)
    layout = layout_graph(graph)
    img = render_to_image(layout, config=render_config)

    return image_to_terminal(img, width, protocol)
