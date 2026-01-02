"""Command-line interface for termflow.

This module provides the CLI entry point for termflow, accessible via:
- `tf` command (when installed)
- `python -m termflow`

Examples:
    $ cat README.md | tf
    $ tf document.md
    $ tf --width 100 --style dracula document.md
    $ echo '# Hello' | tf
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import TextIO

from termflow import __version__
from termflow.config import Config
from termflow.parser import Parser
from termflow.render import Renderer, RenderStyle
from termflow.syntax import Highlighter


def get_terminal_width() -> int:
    """Get terminal width with fallback.

    Returns:
        Terminal width in columns, or 80 if detection fails.
    """
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="tf",
        description="termflow - A streaming markdown renderer for modern terminals",
        epilog="Examples:\n"
        "  cat README.md | tf\n"
        "  tf document.md\n"
        "  tf --width 100 document.md\n"
        "  tf --style dracula README.md\n"
        '  echo "# Hello" | tf',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Markdown file to render (reads stdin if not provided)",
    )

    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=None,
        metavar="N",
        help="Terminal width (default: auto-detect)",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to config file",
    )

    parser.add_argument(
        "--style",
        choices=["default", "dracula", "nord", "gruvbox"],
        default="default",
        help="Color style preset (default: default)",
    )

    parser.add_argument(
        "--syntax-style",
        default=None,
        metavar="NAME",
        help="Pygments syntax highlighting style (monokai, dracula, etc.)",
    )

    parser.add_argument(
        "--list-syntax-styles",
        action="store_true",
        help="List available syntax highlighting styles and exit",
    )

    parser.add_argument(
        "--no-clipboard",
        action="store_true",
        help="Disable OSC 52 clipboard for code blocks",
    )

    parser.add_argument(
        "--no-hyperlinks",
        action="store_true",
        help="Disable OSC 8 hyperlinks",
    )

    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Disable pretty code block borders",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def list_syntax_styles() -> None:
    """Print available syntax highlighting styles."""
    styles = Highlighter.available_styles()
    print("Available syntax highlighting styles:")
    print()

    # Print in columns
    col_width = max(len(s) for s in styles) + 2
    cols = max(1, 80 // col_width)

    for i, style in enumerate(styles):
        print(f"  {style:<{col_width}}", end="")
        if (i + 1) % cols == 0:
            print()

    if len(styles) % cols != 0:
        print()


def stream_render(
    input_stream: TextIO,
    output_stream: TextIO,
    width: int,
    style: RenderStyle,
    config: Config,
) -> None:
    """Stream-render markdown from input to output.

    Args:
        input_stream: Input file-like object.
        output_stream: Output file-like object.
        width: Terminal width.
        style: Render style.
        config: Configuration.
    """
    parser = Parser()
    highlighter = Highlighter(style=config.syntax_style)

    renderer = Renderer(
        output=output_stream,
        width=width,
        style=style,
        features=config.features,
        highlighter=highlighter,
    )

    try:
        for line in input_stream:
            # Strip trailing newline for parsing
            line = line.rstrip("\n\r")
            events = parser.parse_line(line)
            renderer.render_all(events)

        # Finalize any open blocks
        final_events = parser.finalize()
        renderer.render_all(final_events)

    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        print()  # Newline after ^C
    except BrokenPipeError:
        # Clean exit when piped to head, less -F, etc.
        # Prevent broken pipe error message
        with contextlib.suppress(BrokenPipeError):
            sys.stdout.close()


def render_file(
    file_path: Path,
    output_stream: TextIO,
    width: int,
    style: RenderStyle,
    config: Config,
) -> int:
    """Render a markdown file.

    Args:
        file_path: Path to markdown file.
        output_stream: Output file-like object.
        width: Terminal width.
        style: Render style.
        config: Configuration.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        with file_path.open(encoding="utf-8") as f:
            stream_render(f, output_stream, width, style, config)
        return 0
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1
    except PermissionError:
        print(f"Error: Permission denied: {file_path}", file=sys.stderr)
        return 1
    except IsADirectoryError:
        print(f"Error: Is a directory: {file_path}", file=sys.stderr)
        return 1
    except UnicodeDecodeError:
        print(f"Error: Cannot decode file (not UTF-8): {file_path}", file=sys.stderr)
        return 1


def get_style(style_name: str, config: Config) -> RenderStyle:
    """Get RenderStyle from name.

    Args:
        style_name: Style preset name.
        config: Configuration (for default style).

    Returns:
        RenderStyle instance.
    """
    if style_name == "dracula":
        return RenderStyle.dracula()
    elif style_name == "nord":
        return RenderStyle.nord()
    elif style_name == "gruvbox":
        return RenderStyle.gruvbox()
    else:
        return config.style


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code.
    """
    arg_parser = create_parser()
    args = arg_parser.parse_args(argv)

    # Handle --list-syntax-styles
    if args.list_syntax_styles:
        list_syntax_styles()
        return 0

    # Load config
    config = Config.load(args.config)

    # Override config with CLI args
    if args.no_clipboard:
        config.features.clipboard = False
    if args.no_hyperlinks:
        config.features.hyperlinks = False
    if args.no_pretty:
        config.features.pretty_pad = False
    if args.syntax_style:
        config.syntax_style = args.syntax_style

    # Get style
    style = get_style(args.style, config)

    # Determine width
    width = args.width or config.width or get_terminal_width()
    if config.max_width:
        width = min(width, config.max_width)

    # Render from file or stdin
    if args.file:
        return render_file(args.file, sys.stdout, width, style, config)
    else:
        # Check if stdin is a TTY (no input piped)
        if sys.stdin.isatty():
            print("Usage: tf <file> or pipe markdown to tf", file=sys.stderr)
            print("       tf --help for more options", file=sys.stderr)
            return 1

        stream_render(sys.stdin, sys.stdout, width, style, config)
        return 0


if __name__ == "__main__":
    sys.exit(main())
