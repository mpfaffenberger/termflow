"""Tests for termflow.config module."""

import os
import tempfile
from pathlib import Path

from termflow.config import Config, get_config_path, get_default_config


class TestConfig:
    """Test Config class."""

    def test_default_values(self):
        config = Config()
        assert config.width is None
        assert config.max_width == 120
        assert config.syntax_style == "monokai"

    def test_default_style(self):
        config = Config()
        assert config.style is not None
        assert config.style.bright.startswith("#")

    def test_default_features(self):
        config = Config()
        assert config.features is not None
        assert config.features.clipboard is True
        assert config.features.hyperlinks is True
        assert config.features.pretty_pad is True

    def test_to_dict(self):
        config = Config()
        data = config.to_dict()
        assert "width" in data
        assert "max_width" in data
        assert "syntax_style" in data
        assert "style" in data
        assert "features" in data


class TestConfigLoading:
    """Test loading config from TOML files."""

    def test_load_returns_config(self):
        config = Config.load()
        assert isinstance(config, Config)

    def test_load_nonexistent_path(self):
        config = Config.load("/nonexistent/path/config.toml")
        # Should return default config
        assert isinstance(config, Config)
        assert config.max_width == 120

    def test_load_from_valid_toml(self):
        toml_content = """
max_width = 100
syntax_style = "dracula"

[style]
bright = "#ff0000"

[features]
clipboard = false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            try:
                config = Config.load(f.name)
                assert config.max_width == 100
                assert config.syntax_style == "dracula"
                assert config.style.bright == "#ff0000"
                assert config.features.clipboard is False
            finally:
                Path(f.name).unlink()

    def test_load_partial_config(self):
        """Test loading config with only some values set."""
        toml_content = """
max_width = 80
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            try:
                config = Config.load(f.name)
                assert config.max_width == 80
                # Other values should be defaults
                assert config.syntax_style == "monokai"
            finally:
                Path(f.name).unlink()

    def test_load_invalid_toml(self):
        """Test loading invalid TOML gracefully."""
        toml_content = "this is not valid [ toml"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            try:
                config = Config.load(f.name)
                # Should return default config
                assert isinstance(config, Config)
            finally:
                Path(f.name).unlink()


class TestConfigEnvVar:
    """Test TERMFLOW_CONFIG environment variable."""

    def test_env_var_path(self):
        toml_content = """
max_width = 50
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            try:
                old_env = os.environ.get("TERMFLOW_CONFIG")
                os.environ["TERMFLOW_CONFIG"] = f.name

                config = Config.load()
                assert config.max_width == 50
            finally:
                if old_env:
                    os.environ["TERMFLOW_CONFIG"] = old_env
                elif "TERMFLOW_CONFIG" in os.environ:
                    del os.environ["TERMFLOW_CONFIG"]
                Path(f.name).unlink()


class TestGetDefaultConfig:
    """Test get_default_config function."""

    def test_returns_config(self):
        config = get_default_config()
        assert isinstance(config, Config)

    def test_is_default(self):
        config = get_default_config()
        default = Config()
        assert config.max_width == default.max_width
        assert config.syntax_style == default.syntax_style


class TestGetConfigPath:
    """Test get_config_path function."""

    def test_returns_path_or_none(self):
        result = get_config_path()
        assert result is None or isinstance(result, Path)

    def test_respects_env_var(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("max_width = 100")
            f.flush()

            try:
                old_env = os.environ.get("TERMFLOW_CONFIG")
                os.environ["TERMFLOW_CONFIG"] = f.name

                result = get_config_path()
                assert result is not None
                assert str(result) == f.name
            finally:
                if old_env:
                    os.environ["TERMFLOW_CONFIG"] = old_env
                elif "TERMFLOW_CONFIG" in os.environ:
                    del os.environ["TERMFLOW_CONFIG"]
                Path(f.name).unlink()


class TestConfigFromDict:
    """Test Config._from_dict method."""

    def test_basic_values(self):
        data = {
            "max_width": 100,
            "syntax_style": "nord",
        }
        config = Config._from_dict(data)
        assert config.max_width == 100
        assert config.syntax_style == "nord"

    def test_style_section(self):
        data = {
            "style": {
                "bright": "#aabbcc",
                "head": "#ddeeff",
            }
        }
        config = Config._from_dict(data)
        assert config.style.bright == "#aabbcc"
        assert config.style.head == "#ddeeff"

    def test_features_section(self):
        data = {
            "features": {
                "clipboard": False,
                "hyperlinks": False,
            }
        }
        config = Config._from_dict(data)
        assert config.features.clipboard is False
        assert config.features.hyperlinks is False

    def test_empty_dict(self):
        config = Config._from_dict({})
        # Should return default values
        assert config.max_width == 120
