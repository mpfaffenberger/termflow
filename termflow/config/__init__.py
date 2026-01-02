"""Configuration module for termflow.

Provides TOML-based configuration loading with sensible defaults.

Example:
    >>> from termflow.config import Config
    >>> config = Config.load()  # Auto-find config file
    >>> config = Config.load("/path/to/config.toml")  # Explicit path
"""

from termflow.config.config import (
    Config,
    get_config_dir,
    get_config_path,
    get_default_config,
)

__all__ = [
    "Config",
    "get_config_dir",
    "get_config_path",
    "get_default_config",
]
