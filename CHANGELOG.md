# Changelog

All notable changes to termflow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.4] - 2026-08-22

### Added

- `RenderStyle.from_palette()` — derive rendering accents from a
  terminal palette's ANSI slots (Palette object or plain dict), so
  menus and markdown output stay on-theme when the app has remapped
  terminal colors via OSC instead of using the hardcoded defaults.

## [0.2.3] - 2026-08-22

### Added

- `termflow.tui.terminal_session()` — refcounted, thread-safe context
  manager holding raw mode + ONE alternate screen across multiple TUI
  components. Chained menu flows (a picker opening sub-pickers) run each
  menu with `alt_screen=False` inside a session and never flash the
  primary screen between menus. Nested sessions share the outermost one.

## [0.2.2] - 2026-08-22

### Fixed

- `read_key` now reads straight from the file descriptor with `os.read`
  on POSIX ttys. The previous `stream.read(1)` + `select()` combination
  was a trap: `TextIOWrapper` slurped a whole escape-sequence burst into
  its internal buffer while returning one char, `select()` then reported
  nothing pending, and every arrow key parsed as a lone ESC — cancelling
  the menu. Includes UTF-8 multibyte handling and a buffered-stream
  fallback for tests.

## [0.2.1] - 2026-08-22

### Added

- `Menu` / `MenuBuilder` grew the hooks real-world pickers need
  (driven by porting Code Puppy's TUI menus):
  - `.on_key(key, handler)` — custom action keys (pin/clone/delete
    patterns); handlers receive the live `Menu` + highlighted item and
    either exit with a `MenuResult` or mutate state and repaint.
  - `.initial_index(i)` — open with the cursor on the current selection.
  - `.list_width(n)` — fixed left-column width for preview layouts.
  - `.filter_fn(matches)` — custom search matching (fuzzy filters).
  - `Menu.replace_items()`, `Menu.clear_search()`, `Menu.page_up()`,
    `Menu.page_down()`, `Menu.highlighted` — public state surface for
    key handlers.

## [0.2.0] - 2026-08-22

### Added

- `termflow.stream` — steady-rate smooth streaming (ported from Code Puppy):
  - `SteadyDrainer`: adaptive-rate drain engine that buffers bursty producer
    output (LLM token streams) and releases it smoothly via a background
    asyncio task, with pause/abort/close semantics and an injectable
    `is_paused` hook.
  - `StreamSmoother`: plain-text smoother emitting through a callback.
  - `SmoothWriter`: file-like proxy that types pre-rendered ANSI text out
    one visible character at a time, keeping escape sequences atomic.
  - `split_by_visible`: ANSI-aware visible-budget string splitting.
- `termflow.themes` — terminal-level theming:
  - `TerminalPalette` model plus 14 bundled palettes (Catppuccin Mocha/Latte,
    Tokyo Night, Green Screen, Deep Black, Solarized Light, GitHub Light,
    Rose Pine Dawn, Ocean, Forest, Sunset, Vaporwave, Purple Puppy,
    Bubblegum Pink).
  - OSC 4/10/11 palette engine (`apply_palette`, `reset_palette`) with
    best-effort terminal restore at process exit.
  - `TerminalPalette.to_render_style()` bridge deriving a markdown
    `RenderStyle` from a palette.
- `termflow.tui` — dependency-free terminal UI toolkit (no curses, no
  prompt_toolkit):
  - `raw_mode` / `alt_screen` context managers and cursor helpers.
  - `read_key` / `parse_key` keyboard input with full escape-sequence
    parsing (arrows, paging, ctrl-keys) and a Windows `msvcrt` path.
  - `Menu` / `MenuBuilder`: declarative interactive menus with search
    filtering, pagination, multi-select, disabled rows, preview pane,
    highlight callbacks, and fully injectable I/O for testing.

### Changed

- Rewrote the README: corrected package name in install instructions
  (`termflow-md`), fixed repository URLs, documented the new `stream`,
  `themes`, and `tui` subsystems.

## [0.1.11] - 2026-04-18

### Fixed

- `wrap_ansi` now wraps at word boundaries instead of chopping words
  mid-character. A word is only character-wrapped when it is, by itself,
  longer than the available width. This fixes ugly cell wrapping in
  tables (e.g. "bananas" no longer becomes "bana / nas").

## [0.1.10] - 2026-04-18

### Fixed

- Tables now respect the terminal width by default — borders no longer wrap
  past the terminal edge. Column widths are capped proportionally and cell
  content wraps inside the cell instead. `TERMFLOW_MAX_TABLE_WIDTH` still
  works as an optional tighter cap.

## [0.1.3] - 2025-05-25

### Fixed

- Inline code in tables now renders without literal backticks (just styled content)

## [0.1.0] - 2025-01-XX

### Added

- Initial release of termflow 🌊
- Streaming markdown parser with event-based architecture
- Terminal renderer with ANSI true-color (24-bit) support
- Syntax highlighting via Pygments (100+ languages supported)
- CLI tool (`tf`) with streaming support
- Configuration via TOML files

#### Markdown Support

- **Headings** (H1-H6) with distinct visual styles
  - H1: Centered, bold, bright color with double-line underline
  - H2: Bold, bright color with underline
  - H3-H6: Progressively subtle styling
- **Code blocks** with:
  - Unicode box drawing borders (╭╮╰╯│─)
  - Language labels
  - Syntax highlighting
  - OSC 52 clipboard integration
- **Inline code** with background highlighting
- **Text formatting**: bold, italic, underline, strikethrough
- **Lists**:
  - Bullet lists with cycling bullets (• ◦ ▪ ▫ ▸ ▹)
  - Ordered lists with multiple styles (1. a) i. A))
  - Nested lists with proper indentation
- **Tables** with Unicode box drawing borders
- **Block quotes** with vertical bar prefix
- **Think blocks** for LLM chain-of-thought (`<think>...</think>`)
- **Horizontal rules**
- **Links** with OSC 8 hyperlink support
- **Images** (displayed as alt text with 🖼 icon)
- **Footnotes**

#### CLI Features

- `tf <file>` - Render markdown file
- `cat file.md | tf` - Pipe markdown input
- `-w, --width` - Set terminal width
- `--style` - Choose color preset (default, dracula, nord, gruvbox)
- `--syntax-style` - Choose Pygments syntax style
- `--no-clipboard` - Disable OSC 52 clipboard
- `--no-hyperlinks` - Disable OSC 8 links
- `--no-pretty` - Disable decorative borders
- `--list-syntax-styles` - List available syntax styles

#### Configuration

- TOML configuration file support
- Search order: `$TERMFLOW_CONFIG` → `~/.config/termflow/config.toml` → `~/.termflow.toml`
- Customizable colors, features, and syntax style

#### Style Presets

- **default**: Soft, readable colors for dark backgrounds
- **dracula**: Purple-tinted dark theme
- **nord**: Arctic, bluish color scheme
- **gruvbox**: Warm, retro color scheme

[Unreleased]: https://github.com/username/termflow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/username/termflow/releases/tag/v0.1.0
