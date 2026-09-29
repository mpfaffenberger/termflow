"""Whole-document convenience renderers.

One-call helpers on top of :class:`~termflow.render.renderer.Renderer`:

* :func:`render_markdown` -- render a complete document to a stream.
* :func:`render_streaming` -- render lines as an iterator yields them.
* :func:`render_markdown_lines` -- render to a list of display lines at
  a fixed width. Call it again with a new width to *reflow* a document
  after a terminal resize (this is what the markdown pager does).
"""

from __future__ import annotations

import io
from dataclasses import replace
from typing import TYPE_CHECKING, TextIO

from termflow.parser import Parser
from termflow.render.renderer import Renderer
from termflow.render.style import RenderFeatures

if TYPE_CHECKING:
    from collections.abc import Iterable

    from termflow.render.style import RenderStyle
    from termflow.syntax import Highlighter


def render_markdown(
    markdown: str,
    width: int | None = None,
    output: TextIO | None = None,
    style: RenderStyle | None = None,
) -> None:
    """Convenience function to render markdown to terminal.

    Args:
        markdown: Markdown text to render.
        width: Terminal width (tracks the live terminal width if None).
        output: Output stream (stdout if None).
        style: Render style (default if None).

    Example:
        >>> render_markdown("# Hello\\n\\nThis is **bold**!")
    """
    renderer = Renderer(output=output, width=width, style=style)
    renderer.render_all(Parser().parse_document(markdown))


def render_streaming(
    lines_iter: Iterable[str],
    width: int | None = None,
    output: TextIO | None = None,
    style: RenderStyle | None = None,
) -> None:
    """Render markdown from a streaming source.

    Args:
        lines_iter: Iterator/generator yielding lines.
        width: Terminal width (tracks the live terminal width if None).
        output: Output stream (stdout if None).
        style: Render style (default if None).

    Example:
        >>> def stream():
        ...     yield "# Hello"
        ...     yield ""
        ...     yield "World!"
        >>> render_streaming(stream())
    """
    parser = Parser()
    renderer = Renderer(output=output, width=width, style=style)

    for line in lines_iter:
        renderer.render_all(parser.parse_line(line))
    renderer.render_all(parser.finalize())


def render_markdown_lines(
    markdown: str,
    width: int,
    style: RenderStyle | None = None,
    features: RenderFeatures | None = None,
    highlighter: Highlighter | None = None,
) -> list[str]:
    """Render a complete document to display lines at a fixed ``width``.

    Pure: nothing is written anywhere. OSC 52 clipboard copies are
    disabled because they are a side effect, not display content -- a
    reflow would otherwise re-copy every code block on each resize.

    Args:
        markdown: Markdown text to render.
        width: Width to wrap to, in columns.
        style: Render style (default if None).
        features: Feature flags (defaults if None; clipboard forced off).
        highlighter: Syntax highlighter (created if None).

    Returns:
        The rendered lines, without trailing blank lines.
    """
    output = io.StringIO()
    renderer = Renderer(
        output=output,
        width=width,
        style=style,
        features=replace(features or RenderFeatures(), clipboard=False),
        highlighter=highlighter,
    )
    renderer.render_all(Parser().parse_document(markdown))
    return output.getvalue().rstrip("\n").split("\n")
