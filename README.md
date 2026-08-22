# termflow

**A streaming markdown renderer and terminal UI toolkit for modern terminals.**

[![PyPI version](https://img.shields.io/pypi/v/termflow-md.svg)](https://pypi.org/project/termflow-md/)
[![Python versions](https://img.shields.io/pypi/pyversions/termflow-md.svg)](https://pypi.org/project/termflow-md/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

termflow renders markdown to ANSI as it arrives, line by line, which makes it
a natural fit for LLM output. It also ships the surrounding machinery a
terminal-native application needs: smooth typewriter-style output pacing,
terminal-wide color theming, and dependency-free interactive menus.

Two runtime dependencies: Pygments and wcwidth. No curses, no prompt_toolkit,
no Rich.

## Features

- **Streaming rendering** — parse and render markdown incrementally, line by
  line, without waiting for the full document
- **Syntax highlighting** — fenced code blocks highlighted via Pygments, with
  language detection
- **GitHub-flavored tables**, ordered/unordered/nested lists, block quotes,
  and `<think>` blocks for LLM chain-of-thought
- **Smooth output pacing** (`termflow.stream`) — adaptive-rate buffering that
  turns bursty token streams into steady typewriter output
- **Terminal theming** (`termflow.themes`) — bundled 16-color palettes applied
  terminal-wide via OSC escape sequences, with automatic restore on exit
- **Interactive menus** (`termflow.tui`) — a declarative menu builder with
  search, pagination, multi-select, and live preview panes, built on plain
  ANSI escape codes
- **OSC 8 hyperlinks** and **OSC 52 clipboard** integration where the
  terminal supports them
- **Configurable** via TOML config file or programmatic API

## Installation

```bash
pip install termflow-md
```

Or run the CLI directly:

```bash
uvx --from termflow-md tf README.md
```

## CLI

```bash
tf README.md                  # render a file
echo "# Hello" | tf           # render stdin
tf -w 100 document.md         # fixed width
tf --style dracula README.md  # color preset
tf --syntax-style nord doc.md # Pygments style for code blocks
tf --list-syntax-styles       # available syntax styles
```

Run `tf --help` for the full option list.

## Rendering markdown

```python
from termflow import render_markdown

render_markdown("# Hello World")
```

Streaming, the primary use case:

```python
import sys
from termflow import Parser, Renderer

parser = Parser()
renderer = Renderer(output=sys.stdout, width=80)

for line in markdown_stream:
    renderer.render_all(parser.parse_line(line))

renderer.render_all(parser.finalize())
```

Custom styling:

```python
from termflow import Renderer, RenderStyle, RenderFeatures

style = RenderStyle.dracula()  # or .nord(), .gruvbox(), .default()
style = RenderStyle(bright="#87ceeb")  # or roll your own

renderer = Renderer(
    width=100,
    style=style,
    features=RenderFeatures(clipboard=True, hyperlinks=True),
)
```

## Smooth streaming output

Token streams arrive in bursts; printing each chunk immediately makes output
stutter. `termflow.stream` buffers incoming text and drains it at an adaptive
rate from a background asyncio task: latency stays low when the producer runs
hot, and output stays smooth when it trickles.

`SmoothWriter` is a file-like proxy that sits between a `Renderer` (or any
producer of ANSI text) and the real output stream. Escape sequences are
emitted atomically, so styling never tears mid-sequence:

```python
import sys
from termflow import Parser, Renderer
from termflow.stream import SmoothWriter

writer = SmoothWriter(sys.stdout)
writer.start()

renderer = Renderer(output=writer, width=80)
parser = Parser()
async for chunk in model_stream:
    renderer.render_all(parser.parse_line(chunk))

await writer.close()  # waits for the buffer to finish draining
# writer.abort()       # or: stop typing NOW and drop the backlog
```

`StreamSmoother` does the same for plain text via an emit callback, and both
accept an `is_paused` hook to hold output while something else owns the
terminal.

## Terminal theming

`termflow.themes` recolors the whole terminal window — background, foreground,
and the 16 ANSI palette slots — using xterm OSC sequences supported by iTerm2,
Terminal.app, kitty, Alacritty, VS Code, GNOME Terminal, and Windows Terminal.
Unsupported terminals ignore them silently. An atexit handler restores the
terminal on process exit.

```python
from termflow.themes import PALETTES, apply_palette, reset_palette

apply_palette(PALETTES["catppuccin_mocha"])
reset_palette()  # back to the terminal's own colors
```

Bundled palettes: Catppuccin Mocha/Latte, Tokyo Night, Solarized Light,
GitHub Light, Rose Pine Dawn, and a set of originals (ocean, forest, sunset,
vaporwave, green_screen, deep_black, purple_puppy, bubblegum_pink).

Each palette bridges to the markdown renderer, so themed output matches the
terminal chrome:

```python
from termflow import Renderer
from termflow.themes import get_palette

palette = get_palette("tokyo_night")
renderer = Renderer(style=palette.to_render_style())
```

## Interactive menus

`termflow.tui` provides a menu component built on raw ANSI escape codes:
alternate screen, arrow-key navigation, incremental search, pagination,
multi-select, and a live preview pane. Every I/O surface (key source, output
stream, terminal size) is injectable, so menus are testable without a tty.

```python
from termflow.tui import MenuBuilder, MenuItem

result = (
    MenuBuilder("Pick a model")
    .items(
        [
            MenuItem("gpt-5", description="fast and smart"),
            MenuItem("claude", description="thoughtful"),
            MenuItem("qwen", description="local"),
        ]
    )
    .searchable()
    .page_size(10)
    .preview(lambda item: f"Details for {item.label}")
    .run()
)

if not result.cancelled:
    print(result.item.value)
```

Multi-select returns `result.items`; `on_highlight` fires on every cursor
move (useful for live theme previews); disabled items render dim and are
skipped by navigation.

## Configuration

Create `~/.config/termflow/config.toml` (or point `TERMFLOW_CONFIG` at a
path):

```toml
width = 0            # 0 = auto-detect
max_width = 120
syntax_style = "monokai"

[style]
bright = "#87ceeb"   # main accent (H1/H2)
head = "#98fb98"     # H3
symbol = "#dda0dd"   # bullets, borders
link = "#87cefa"
error = "#ff6b6b"

[features]
clipboard = true     # OSC 52 clipboard for code blocks
hyperlinks = true    # OSC 8 clickable links
pretty_pad = true    # unicode borders on code blocks
```

See `examples/config.toml` for the full set of options.

## Origin

termflow began as a Python port of
[streamdown-rs](https://github.com/streamdown-rs/streamdown), a streaming
markdown renderer written in Rust, and has since grown into a broader
terminal UI toolkit.

## Contributing

```bash
git clone https://github.com/mpfaffenberger/termflow.git
cd termflow
pip install -e ".[dev]"

pytest tests/ -v
ruff check .
ruff format .
```

Pull requests are welcome.

## License

MIT. See [LICENSE](LICENSE).
