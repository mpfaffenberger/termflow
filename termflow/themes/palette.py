"""Bundled terminal palettes and the TerminalPalette model.

Each palette ships:

* ``bg``   -- #rrggbb default background
* ``fg``   -- #rrggbb default foreground
* ``ansi`` -- 16 #rrggbb hex strings (ANSI slots 0-15)

The Catppuccin / Tokyo Night / Solarized palettes follow the canonical
upstream specs. The rest are original, coherent schemes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from termflow.ansi.color import hex2rgb, rgb2hex
from termflow.render.style import RenderStyle


def blend_hex(a: str, b: str, t: float) -> str:
    """Linearly blend two hex colors: ``t=0`` gives ``a``, ``t=1`` gives ``b``.

    Falls back to ``a`` when either color fails to parse.

    Example:
        >>> blend_hex("#000000", "#ffffff", 0.5)
        '#7F7F7F'
    """
    ra = hex2rgb(a)
    rb = hex2rgb(b)
    if ra is None or rb is None:
        return a
    t = max(0.0, min(1.0, t))
    return rgb2hex(*(round(ca + (cb - ca) * t) for ca, cb in zip(ra, rb, strict=True)))


@dataclass(frozen=True)
class TerminalPalette:
    """A full 16-color terminal palette plus default bg/fg.

    Attributes:
        name: Registry key / display identifier.
        bg: Default background color (#rrggbb).
        fg: Default foreground color (#rrggbb).
        ansi: The 16 ANSI slot colors (0-15).
    """

    name: str
    bg: str
    fg: str
    ansi: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.ansi) != 16:
            raise ValueError(f"palette {self.name!r} needs 16 ANSI colors, got {len(self.ansi)}")

    @classmethod
    def from_dict(cls, name: str, data: dict) -> TerminalPalette:
        """Build a palette from the ``{"bg", "fg", "ansi"}`` dict shape."""
        return cls(name=name, bg=data["bg"], fg=data["fg"], ansi=tuple(data["ansi"]))

    def to_dict(self) -> dict:
        """Serialize to the ``{"bg", "fg", "ansi"}`` dict shape."""
        return {"bg": self.bg, "fg": self.fg, "ansi": list(self.ansi)}

    def to_render_style(self) -> RenderStyle:
        """Derive a markdown :class:`RenderStyle` from this palette.

        Accents come from the bright ANSI slots; the ``dark``/``mid``/
        ``light`` background tiers are blends of bg toward fg so code
        blocks and tables sit naturally on the themed background.
        """
        return RenderStyle(
            bright=self.ansi[12],  # bright blue - H1/H2
            head=self.ansi[10],  # bright green - H3
            symbol=self.ansi[13],  # bright magenta - markers, borders
            grey=self.ansi[8],  # bright black - dim text
            dark=blend_hex(self.bg, self.fg, 0.08),
            mid=blend_hex(self.bg, self.fg, 0.16),
            light=blend_hex(self.bg, self.fg, 0.24),
            link=self.ansi[14],  # bright cyan - links
            error=self.ansi[9],  # bright red - errors
        )


# =============================================================================
# Bundled palettes
# =============================================================================

_RAW_PALETTES: dict[str, dict] = {
    "catppuccin_mocha": {
        "bg": "#1e1e2e",
        "fg": "#cdd6f4",
        "ansi": [
            "#45475a",
            "#f38ba8",
            "#a6e3a1",
            "#f9e2af",
            "#89b4fa",
            "#f5c2e7",
            "#94e2d5",
            "#bac2de",
            "#585b70",
            "#f38ba8",
            "#a6e3a1",
            "#f9e2af",
            "#89b4fa",
            "#f5c2e7",
            "#94e2d5",
            "#a6adc8",
        ],
    },
    "catppuccin_latte": {
        "bg": "#eff1f5",
        "fg": "#4c4f69",
        "ansi": [
            "#4c4f69",
            "#d20f39",
            "#40a02b",
            "#df8e1d",
            "#1e66f5",
            "#8839ef",
            "#179299",
            "#acb0be",
            "#5c5f77",
            "#e64553",
            "#a6d189",
            "#fe640b",
            "#7287fd",
            "#ea76cb",
            "#94e2d5",
            "#bcc0cc",
        ],
    },
    "tokyo_night": {
        "bg": "#1a1b26",
        "fg": "#c0caf5",
        "ansi": [
            "#15161e",
            "#f7768e",
            "#9ece6a",
            "#e0af68",
            "#7aa2f7",
            "#bb9af7",
            "#7dcfff",
            "#a9b1d6",
            "#414868",
            "#f7768e",
            "#9ece6a",
            "#e0af68",
            "#7aa2f7",
            "#bb9af7",
            "#7dcfff",
            "#c0caf5",
        ],
    },
    "green_screen": {
        # Black glass, green phosphor, and one intentionally
        # eye-searing highlight.
        "bg": "#000000",
        "fg": "#6a9955",
        "ansi": [
            "#000000",
            "#6a9955",
            "#6a9955",
            "#6a9955",
            "#6a9955",
            "#6a9955",
            "#6a9955",
            "#6a9955",
            "#3a5945",
            "#6a9955",
            "#00ff00",
            "#6a9955",
            "#6a9955",
            "#6a9955",
            "#6a9955",
            "#00ff00",
        ],
    },
    "deep_black": {
        "bg": "#050505",
        "fg": "#e6e6e6",
        "ansi": [
            "#050505",
            "#ff6b6b",
            "#94d82d",
            "#ffd166",
            "#4dabf7",
            "#b197fc",
            "#63e6be",
            "#adb5bd",
            "#1f1f1f",
            "#ff8787",
            "#a9e34b",
            "#ffe066",
            "#74c0fc",
            "#d0bfff",
            "#96f2d7",
            "#f1f3f5",
        ],
    },
    "solarized_light": {
        "bg": "#fdf6e3",
        "fg": "#657b83",
        "ansi": [
            "#073642",
            "#dc322f",
            "#859900",
            "#b58900",
            "#268bd2",
            "#d33682",
            "#2aa198",
            "#eee8d5",
            "#fdf6e3",
            "#cb4b16",
            "#93a1a1",
            "#839496",
            "#657b83",
            "#6c71c4",
            "#586e75",
            "#002b36",
        ],
    },
    "github_light": {
        "bg": "#ffffff",
        "fg": "#24292e",
        "ansi": [
            "#24292e",
            "#d73a49",
            "#28a745",
            "#dbab09",
            "#0366d6",
            "#6f42c1",
            "#1b7c83",
            "#6a737d",
            "#586069",
            "#cb2431",
            "#22863a",
            "#b08800",
            "#005cc5",
            "#5a32a3",
            "#3192aa",
            "#d1d5da",
        ],
    },
    "rose_pine_dawn": {
        "bg": "#faf4ed",
        "fg": "#575279",
        "ansi": [
            "#f2e9e1",
            "#b4637a",
            "#56949f",
            "#ea9d34",
            "#286983",
            "#907aa9",
            "#d7827e",
            "#575279",
            "#9893a5",
            "#b4637a",
            "#56949f",
            "#ea9d34",
            "#286983",
            "#907aa9",
            "#d7827e",
            "#cecacd",
        ],
    },
    "ocean": {
        "bg": "#0a1929",
        "fg": "#d6eaf8",
        "ansi": [
            "#0a1929",
            "#e74c3c",
            "#48c9b0",
            "#f4d03f",
            "#3498db",
            "#1abc9c",
            "#5dade2",
            "#aed6f1",
            "#34495e",
            "#ec7063",
            "#1abc9c",
            "#f7dc6f",
            "#5499c7",
            "#48c9b0",
            "#85c1e9",
            "#ebf5fb",
        ],
    },
    "forest": {
        "bg": "#1a2310",
        "fg": "#e3eecc",
        "ansi": [
            "#1a2310",
            "#c0392b",
            "#27ae60",
            "#d4ac0d",
            "#7d6608",
            "#16a085",
            "#1e8449",
            "#aed581",
            "#52682d",
            "#cd6155",
            "#52be80",
            "#f4d03f",
            "#7d6608",
            "#48c9b0",
            "#7dcea0",
            "#eaf2cf",
        ],
    },
    "sunset": {
        "bg": "#2d1b0e",
        "fg": "#ffe4cc",
        "ansi": [
            "#2d1b0e",
            "#e74c3c",
            "#d35400",
            "#f39c12",
            "#7d3c98",
            "#c0392b",
            "#e67e22",
            "#fad7a0",
            "#5d4037",
            "#ec7063",
            "#e67e22",
            "#f9e79f",
            "#a93226",
            "#d35400",
            "#f5b041",
            "#fdebd0",
        ],
    },
    "vaporwave": {
        "bg": "#16002a",
        "fg": "#ffe0ff",
        "ansi": [
            "#16002a",
            "#ff6ec7",
            "#48c9b0",
            "#f7dc6f",
            "#bb6bd9",
            "#ec407a",
            "#7fdbff",
            "#e8daef",
            "#5b2c6f",
            "#ff79c6",
            "#80deea",
            "#fff59d",
            "#d7bde2",
            "#f06292",
            "#80deea",
            "#fce4ec",
        ],
    },
    "purple_puppy": {
        # Violet fur, lavender muzzle, pink tongue, inky background.
        "bg": "#1c0630",
        "fg": "#f0e3ff",
        "ansi": [
            "#2a0a45",
            "#ff5c8a",
            "#c986ff",
            "#f5c26b",
            "#9b30d9",
            "#ff7fa8",
            "#d9a7f5",
            "#f0e3ff",
            "#a98aca",
            "#ff7fa8",
            "#dba1ff",
            "#ffd58f",
            "#b06be8",
            "#ff9fd2",
            "#e8ccff",
            "#fdf7ff",
        ],
    },
    "bubblegum_pink": {
        "bg": "#2a0f1f",
        "fg": "#fff1f7",
        "ansi": [
            "#2a0f1f",
            "#ff5fa2",
            "#ff8ec7",
            "#ffd1e8",
            "#ff4f93",
            "#d96cff",
            "#ff9fd2",
            "#fff1f7",
            "#4d1d39",
            "#ff7ab2",
            "#ffb3da",
            "#ffe0ef",
            "#ff6fb0",
            "#eb8cff",
            "#ffc4e1",
            "#fff8fb",
        ],
    },
}

#: All bundled palettes, keyed by name.
PALETTES: dict[str, TerminalPalette] = {
    name: TerminalPalette.from_dict(name, data) for name, data in _RAW_PALETTES.items()
}


def get_palette(name: str) -> TerminalPalette | None:
    """Look up a bundled palette by name (None if unknown).

    Lookup is forgiving: case-insensitive, with spaces/hyphens treated
    as underscores (``"Tokyo-Night"`` finds ``tokyo_night``).
    """
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return PALETTES.get(normalized)


def palette_names() -> list[str]:
    """Names of all bundled palettes, in registry order."""
    return list(PALETTES)
