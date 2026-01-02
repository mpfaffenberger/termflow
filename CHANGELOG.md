# Changelog

All notable changes to termflow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
