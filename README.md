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
- **Word wrapping that follows the terminal** — prose wraps at word
  boundaries (with hanging indents for lists), long code lines wrap with a
  `↪` marker, and new output adapts when the terminal is resized
- **Reflowing pager** (`tf --pager`) — a scrollable view that re-wraps the
  whole document on every resize
- **Smooth output pacing** (`termflow.stream`) — adaptive-rate buffering that
  turns bursty token streams into steady typewriter output
- **Terminal theming** (`termflow.themes`) — bundled 16-color palettes applied
  terminal-wide via OSC escape sequences, with automatic restore on exit
- **Interactive menus** (`termflow.tui`) — a declarative menu builder with
  search, pagination, multi-select, and live preview panes, built on plain
  ANSI escape codes
- **Live mode** (`termflow.live`) — real-time, multi-window terminal apps:
  a frame loop with diffed rendering, true key press/release input (Windows
  Terminal included), truecolor half-block pixel windows, and markdown panes.
  `tf --doom` plays a raycaster next to a streaming mission briefing, and
  `tf --doom-shareware` plays the real DOOM
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
tf -w 100 document.md         # fixed width (default: follow the terminal)
tf -p README.md               # pager that re-wraps on resize (stdin works too)
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
renderer = Renderer(output=sys.stdout)  # no width: follows terminal resizes

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

## Resizing

A `Renderer` without a fixed `width` re-checks the terminal width before
every block, so output rendered *after* a resize fits the new size (`max_width`
caps it on very wide terminals). Text already printed to scrollback can't be
reflowed: once termflow emits a newline, the terminal owns that line. If you
need the *whole* document to reflow, use the pager. It keeps the source and
re-renders it at the new width on every resize, keeping your place:

```python
from termflow.tui import PagerBuilder

PagerBuilder("README").markdown(open("README.md").read()).run()

# Or reflow anything: reflow(width) -> lines is re-run when the width changes
PagerBuilder("Log").reflow(lambda width: render_my_lines(width)).run()
```

## Swappable agent histories

`AgentHistory` keeps independent in-memory transcripts for a main agent and its
workers. Give each producer its own file-like buffer (also accepted by
`Renderer(output=...)`), then open a live viewer:

```python
from termflow.tui import AgentHistory, AgentHistoryViewer

history = AgentHistory()
main_output = history.add("main")
worker_output = history.add("worker")
main_output.write("Main agent started\n")
worker_output.write("Worker started\n")

# Producers can keep writing from worker threads while the viewer runs.
AgentHistoryViewer(history).run()
```

Tab / Right cycles forward; Left cycles backward. Each agent retains its scroll
position. New output follows automatically until you scroll away from the bottom;
End / G resumes following. Hidden agents continue collecting output, and agents
registered while the viewer runs join the cycle. `select(agent_id)` switches
programmatically on the viewer's thread; `active_agent` identifies the selection
for host-provided key handlers. All normal pager navigation and close keys apply.

Buffers accept plain text or ANSI-styled, newline-delimited text, not arbitrary
terminal cursor-control output. Lines are clipped to the viewport like a pager,
not reflowed. Histories are retained in memory without a size limit. Buffer writes
and registration are thread-safe; viewer operations belong to its UI thread.
Use `use_alt_screen=False` inside an existing `terminal_session`. Only the viewer
should write to the real terminal while it is open. In async applications, run
the blocking viewer in a worker with coordinated input ownership and cancellation.

This is a viewing/output primitive: routing prompts, steering, cancellation, and
agent lifecycles remain the host application's responsibility. It does not attach
to Code Puppy or CLAI2 automatically.

## Live mode

`termflow.live` runs a frame loop instead of waiting for keys: every frame it
polls input, updates each window, paints a cell buffer, and sends only the
cells that changed since the previous frame. Windows can hold markdown, logs,
or raw RGB framebuffers drawn with `▀` half-blocks (two truecolor pixels per
cell, no graphics protocol needed).

```bash
tf --doom                           # the demo: WASD/arrows, Space fires, Ctrl+Q quits
python -m termflow.live.demos.doom --fps 60
```

```python
from termflow.live import LiveApp, MarkdownView, PixelWidget, TextLog, Window, hsplit, vsplit


class Plasma(PixelWidget):
    t = 0.0

    def update(self, dt, keys):
        self.t += dt * (3 if keys.is_down(" ") else 1)  # held keys, not just presses

    def render(self, surface):
        w, t = surface.width, int(self.t * 60)
        for i in range(len(surface.pixels)):
            surface.pixels[i] = ((i % w + t) & 255) << 16 | ((i // w * 4 + t) & 255)


def layout(area):  # one rect per window
    left, right = hsplit(area, 2, 1)
    return [left, *vsplit(right, 1, 1)]


log = TextLog()  # file-like: Renderer(output=log) works
log.write("hello from the log\n")
windows = [
    Window("Plasma", Plasma()),
    Window("Notes", MarkdownView("# Hi", reveal_rate=60)),
    Window("Log", log),
]
LiveApp(windows, layout).run()  # Tab cycles focus, Ctrl+Q quits
```

Input reports **key releases**, so movement feels like a game instead of
keyboard autorepeat:

- **Windows** (Windows Terminal and conhost): native console input records via
  `ReadConsoleInputW`, including key-up events.
- **POSIX**: raw mode plus the kitty keyboard protocol (kitty, WezTerm, foot,
  Ghostty, recent iTerm2 and Alacritty). Other terminals fall back to
  autorepeat-based holds, which feel a little sticky. The demo's status bar
  shows which input mode is active.

### The real DOOM

```bash
pip install "termflow-md[doom]"      # adds the wasmtime WebAssembly runtime
tf --doom-shareware                  # the shareware episode, Knee-Deep in the Dead
python -m termflow.live.demos.doom_wasm --wad freedoom1.wad   # or your own WADs
```

This runs [doom.wasm](https://github.com/jacobenget/doom.wasm) (doomgeneric
compiled to WebAssembly) inside a `FramebufferView`, letterboxed to 4:3, with
Doom's console output in a side window. termflow (MIT) does not bundle the
engine. On first use it downloads the GPL-2.0 module from its GitHub release
(about 4.6 MB, shareware WAD included), checks it against a pinned SHA-256,
and caches it (`%LOCALAPPDATA%\termflow` on Windows, `~/.cache/termflow`
elsewhere). Vanilla controls apply: arrows move, Ctrl fires, Space opens
doors, Esc opens the menu, Tab shows the automap, and `,` / `.` strafe.
Saving games is not supported yet. For more pixels, zoom the terminal out
(Ctrl+- in Windows Terminal).

**Sound (Windows):** doom.wasm has no audio output, so termflow reads Doom's
own sound bookkeeping from the module's memory. Each `S_sfx` entry's
`usefulness` count goes up when an effect starts, and `S_music` records the
current song. The samples come from the WAD: `DS*` lumps are mixed through
`waveOut`, and `D_*` MUS scores are converted to MIDI for the built-in
Microsoft GS Wavetable Synth. Both use `winmm` via ctypes, so there are no
extra dependencies. Effects play at full volume, because Doom's distance and
panning calculations never reach memory we can read. `--no-sound` turns it
off. Other platforms are silent for now.

Frames are wrapped in synchronized output (DEC mode 2026) to avoid tearing.
Terminals without it ignore the sequence. For an engine that produces its own
frames (an emulator, a native game port, video), push them into a
`FramebufferView` with `set_frame(pixels, width, height)` and they are scaled
to the window. Diffing keeps static panes free. Pixel rows go through a
specialized encoder that uses `▀`, `▄`, or a space depending on which needs
the fewest color changes. DOOM measured in Windows Terminal: about 40 fps
(the cap) up to roughly 225×60, and about 20 fps at 420×110, where the
terminal's parsing speed becomes the limit.

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
