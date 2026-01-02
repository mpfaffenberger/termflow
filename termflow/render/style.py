"""Render style configuration for terminal output.

This module provides color and feature configuration for the renderer.
Colors are specified as hex strings (#RRGGBB) for true-color support.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderStyle:
    """Color and style configuration for rendering.

    All colors are hex strings (#RRGGBB). The default palette uses
    soft, readable colors that work well on dark backgrounds.

    Attributes:
        bright: Bright accent color (H1, H2 headings)
        head: Secondary heading color (H3)
        symbol: Markers, borders, bullets
        grey: Dim text, H6 headings
        dark: Code block background
        mid: Table header background
        light: Light accent
        link: Hyperlink color
        error: Error/warning color
    """

    bright: str = "#87ceeb"  # Sky blue - H1, H2
    head: str = "#98fb98"  # Pale green - H3
    symbol: str = "#dda0dd"  # Plum - markers, borders
    grey: str = "#808080"  # Grey - H6, dim text
    dark: str = "#1a1a2e"  # Dark blue-grey - code bg
    mid: str = "#2d2d44"  # Mid blue-grey - table header bg
    light: str = "#3d3d5c"  # Light blue-grey
    link: str = "#6699cc"  # Steel blue - links
    error: str = "#ff6b6b"  # Coral red - errors

    @classmethod
    def default(cls) -> RenderStyle:
        """Create default style."""
        return cls()

    @classmethod
    def from_hue(cls, hue: float) -> RenderStyle:
        """Generate a style from a base hue (0.0-1.0).

        Creates a cohesive color scheme from a single hue value,
        useful for theming.

        Args:
            hue: Base hue value (0.0 = red, 0.33 = green, 0.66 = blue)

        Returns:
            RenderStyle with generated colors.

        Example:
            >>> style = RenderStyle.from_hue(0.6)  # Blue-ish theme
        """
        from termflow.ansi import hsv_to_rgb, rgb2hex

        def make_color(h_offset: float, s: float, v: float) -> str:
            h = (hue + h_offset) % 1.0
            r, g, b = hsv_to_rgb(h, s, v)
            return rgb2hex(r, g, b)

        return cls(
            bright=make_color(0.0, 0.6, 1.0),
            head=make_color(0.1, 0.4, 0.95),
            symbol=make_color(-0.1, 0.5, 0.85),
            grey=make_color(0.0, 0.0, 0.5),
            dark=make_color(0.0, 0.6, 0.12),
            mid=make_color(0.0, 0.4, 0.25),
            light=make_color(0.0, 0.3, 0.35),
            link=make_color(0.55, 0.5, 0.8),
            error=make_color(0.0, 0.7, 1.0),
        )

    @classmethod
    def dracula(cls) -> RenderStyle:
        """Dracula-inspired color scheme."""
        return cls(
            bright="#bd93f9",  # Purple
            head="#50fa7b",  # Green
            symbol="#ff79c6",  # Pink
            grey="#6272a4",  # Comment
            dark="#282a36",  # Background
            mid="#44475a",  # Current line
            light="#6272a4",  # Comment
            link="#8be9fd",  # Cyan
            error="#ff5555",  # Red
        )

    @classmethod
    def nord(cls) -> RenderStyle:
        """Nord-inspired color scheme."""
        return cls(
            bright="#88c0d0",  # Frost
            head="#a3be8c",  # Aurora green
            symbol="#b48ead",  # Aurora purple
            grey="#4c566a",  # Polar night
            dark="#2e3440",  # Polar night
            mid="#3b4252",  # Polar night
            light="#434c5e",  # Polar night
            link="#81a1c1",  # Frost
            error="#bf616a",  # Aurora red
        )

    @classmethod
    def gruvbox(cls) -> RenderStyle:
        """Gruvbox-inspired color scheme."""
        return cls(
            bright="#fabd2f",  # Yellow
            head="#b8bb26",  # Green
            symbol="#d3869b",  # Purple
            grey="#928374",  # Grey
            dark="#1d2021",  # Dark0 hard
            mid="#282828",  # Dark0
            light="#3c3836",  # Dark1
            link="#83a598",  # Blue
            error="#fb4934",  # Red
        )


@dataclass
class RenderFeatures:
    """Feature flags for rendering behavior.

    Attributes:
        clipboard: Enable OSC 52 clipboard integration for code blocks
        savebrace: Save code blocks to temp files
        pretty_pad: Add decorative box borders to code blocks
        hyperlinks: Enable OSC 8 hyperlinks
        images: Attempt to render images (Kitty/iTerm2 protocol)
        wrap_text: Wrap long lines to terminal width
    """

    clipboard: bool = True
    savebrace: bool = False
    pretty_pad: bool = True
    hyperlinks: bool = True
    images: bool = False
    wrap_text: bool = True
