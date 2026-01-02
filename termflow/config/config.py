"""Configuration loading and management.

This module provides configuration loading from TOML files with
sensible defaults and environment variable overrides.

Search order for config files:
1. Explicit path argument
2. TERMFLOW_CONFIG environment variable
3. ~/.config/termflow/config.toml (XDG)
4. ~/.termflow.toml
5. Default configuration
"""

from __future__ import annotations

import os

# Python 3.11+ has tomllib built-in
import tomllib  # type: ignore[assignment]
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from termflow.render.style import RenderFeatures, RenderStyle


@dataclass
class Config:
    """Main configuration for termflow.

    Loads from TOML config file and environment variables.

    Attributes:
        width: Fixed terminal width (None = auto-detect)
        max_width: Maximum width cap
        style: Color/style configuration
        features: Feature flags
        syntax_style: Pygments style name for syntax highlighting

    Example config.toml:
        ```toml
        max_width = 100
        syntax_style = "dracula"

        [style]
        bright = "#bd93f9"
        head = "#50fa7b"

        [features]
        clipboard = true
        pretty_pad = true
        ```
    """

    # Width settings
    width: int | None = None
    max_width: int = 120

    # Style
    style: RenderStyle = field(default_factory=RenderStyle)

    # Features
    features: RenderFeatures = field(default_factory=RenderFeatures)

    # Syntax highlighting
    syntax_style: str = "monokai"

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        """Load configuration from TOML file.

        Search order:
        1. Explicit path argument
        2. TERMFLOW_CONFIG environment variable
        3. ~/.config/termflow/config.toml (XDG)
        4. ~/.termflow.toml
        5. Default config

        Args:
            path: Explicit path to config file.

        Returns:
            Loaded configuration.
        """
        config_paths: list[Path] = []

        if path:
            config_paths.append(Path(path))

        if env_path := os.environ.get("TERMFLOW_CONFIG"):
            config_paths.append(Path(env_path))

        # XDG config directory
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path("~/.config").expanduser()))
        config_paths.append(Path(xdg_config) / "termflow" / "config.toml")

        # Home dotfile
        config_paths.append(Path.home() / ".termflow.toml")

        for config_path in config_paths:
            if config_path.exists():
                return cls._load_from_file(config_path)

        return cls()

    @classmethod
    def _load_from_file(cls, path: Path) -> Config:
        """Load config from a specific file.

        Args:
            path: Path to TOML config file.

        Returns:
            Loaded configuration, or default if loading fails.
        """
        if tomllib is None:
            # No TOML parser available
            return cls()

        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
            return cls._from_dict(data)
        except Exception:
            # Silently fall back to defaults on any error
            return cls()

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        """Create config from dictionary.

        Args:
            data: Dictionary from TOML parsing.

        Returns:
            Config instance.
        """
        config = cls()

        # Top-level settings
        if "width" in data:
            config.width = int(data["width"])
        if "max_width" in data:
            config.max_width = int(data["max_width"])
        if "syntax_style" in data:
            config.syntax_style = str(data["syntax_style"])

        # Style section [style]
        if style_data := data.get("style"):
            config.style = RenderStyle(
                bright=style_data.get("bright", config.style.bright),
                head=style_data.get("head", config.style.head),
                symbol=style_data.get("symbol", config.style.symbol),
                grey=style_data.get("grey", config.style.grey),
                dark=style_data.get("dark", config.style.dark),
                mid=style_data.get("mid", config.style.mid),
                light=style_data.get("light", config.style.light),
                link=style_data.get("link", config.style.link),
                error=style_data.get("error", config.style.error),
            )

        # Features section [features]
        if features_data := data.get("features"):
            config.features = RenderFeatures(
                clipboard=features_data.get("clipboard", config.features.clipboard),
                savebrace=features_data.get("savebrace", config.features.savebrace),
                pretty_pad=features_data.get("pretty_pad", config.features.pretty_pad),
                hyperlinks=features_data.get("hyperlinks", config.features.hyperlinks),
                images=features_data.get("images", config.features.images),
                wrap_text=features_data.get("wrap_text", config.features.wrap_text),
            )

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary (for saving).

        Returns:
            Dictionary representation.
        """
        return {
            "width": self.width,
            "max_width": self.max_width,
            "syntax_style": self.syntax_style,
            "style": {
                "bright": self.style.bright,
                "head": self.style.head,
                "symbol": self.style.symbol,
                "grey": self.style.grey,
                "dark": self.style.dark,
                "mid": self.style.mid,
                "light": self.style.light,
                "link": self.style.link,
                "error": self.style.error,
            },
            "features": {
                "clipboard": self.features.clipboard,
                "savebrace": self.features.savebrace,
                "pretty_pad": self.features.pretty_pad,
                "hyperlinks": self.features.hyperlinks,
                "images": self.features.images,
                "wrap_text": self.features.wrap_text,
            },
        }


def get_default_config() -> Config:
    """Get default configuration.

    Returns:
        Default Config instance.
    """
    return Config()


def get_config_path() -> Path | None:
    """Find the active config file path, if any.

    Returns:
        Path to existing config file, or None if none found.
    """
    # Check environment variable first
    if env_path := os.environ.get("TERMFLOW_CONFIG"):
        path = Path(env_path)
        if path.exists():
            return path

    # XDG config
    xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path("~/.config").expanduser()))
    xdg_path = Path(xdg_config) / "termflow" / "config.toml"
    if xdg_path.exists():
        return xdg_path

    # Home dotfile
    home_path = Path.home() / ".termflow.toml"
    if home_path.exists():
        return home_path

    return None


def get_config_dir() -> Path:
    """Get the config directory (creating if needed).

    Returns:
        Path to config directory.
    """
    xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path("~/.config").expanduser()))
    config_dir = Path(xdg_config) / "termflow"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
