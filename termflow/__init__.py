"""termflow - A streaming markdown renderer for modern terminals.

A Python port of streamdown-rs, providing beautiful markdown rendering
directly in your terminal with streaming support, syntax highlighting,
and rich formatting.

Basic usage:
    >>> from termflow import render_markdown
    >>> render_markdown("# Hello World!")

Streaming usage:
    >>> from termflow import Parser, Renderer
    >>> parser = Parser()
    >>> renderer = Renderer(width=80)
    >>>
    >>> for line in markdown_lines:
    ...     events = parser.parse_line(line)
    ...     renderer.render_all(events)
    >>>
    >>> renderer.render_all(parser.finalize())

With configuration:
    >>> from termflow import Config, Renderer, RenderStyle
    >>> config = Config.load()
    >>> style = RenderStyle.dracula()
    >>> renderer = Renderer(style=style, features=config.features)

CLI usage:
    $ cat README.md | tf
    $ tf document.md
    $ tf --style dracula README.md
"""

__version__ = "0.1.0"

from termflow.config import Config
from termflow.parser import Parser
from termflow.parser.events import ParseEvent
from termflow.render import Renderer, RenderFeatures, RenderStyle, render_markdown
from termflow.syntax import Highlighter, highlight_code

__all__ = [
    # Core classes
    "Config",
    "Highlighter",
    # Events
    "ParseEvent",
    "Parser",
    # Style
    "RenderFeatures",
    "RenderStyle",
    "Renderer",
    # Version
    "__version__",
    # Convenience functions
    "highlight_code",
    "render_markdown",
]
