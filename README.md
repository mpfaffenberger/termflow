<div align="center">

<!-- Logo placeholder -->
<!-- <img src="assets/logo.png" alt="termflow logo" width="200"/> -->

# termflow 🌊

**A streaming markdown renderer for modern terminals**

[![PyPI version](https://img.shields.io/pypi/v/termflow.svg)](https://pypi.org/project/termflow/)
[![Python versions](https://img.shields.io/pypi/pyversions/termflow.svg)](https://pypi.org/project/termflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build status](https://img.shields.io/github/actions/workflow/status/your-username/termflow/ci.yml?branch=main)](https://github.com/your-username/termflow/actions)

*Perfect for rendering LLM output in real-time.*

</div>

---

## ✨ Features

- **📡 Streaming Rendering** - Render markdown as it arrives, line by line
- **🎨 Syntax Highlighting** - Beautiful code blocks powered by Pygments
- **📊 Tables** - Full support for GitHub-flavored markdown tables
- **📝 Lists** - Ordered, unordered, and nested lists with smart indentation
- **💻 Code Blocks** - Fenced code blocks with language detection and clipboard support
- **💭 Think Blocks** - Special rendering for `<think>` tags (great for LLM chain-of-thought)
- **🔗 Hyperlinks** - OSC 8 clickable links in supported terminals
- **📋 Clipboard** - OSC 52 clipboard integration for code blocks
- **🎛️ Configurable** - Customize colors, styles, and features via TOML config
- **⚡ Fast** - Lightweight and performant, minimal dependencies

## 🚀 Quick Installation

### Using uvx (recommended)

```bash
uvx termflow
```

### Using pip

```bash
pip install termflow-md
```

### From source

```bash
git clone https://github.com/your-username/termflow.git
cd termflow
pip install -e ".[dev]"
```

## 📖 Usage

### Command Line

```bash
# Render a file
tf README.md

# Pipe markdown content
echo "# Hello World" | tf

# Pipe from LLM output
curl -s https://api.example.com/chat | tf

# Set terminal width
tf -w 100 document.md

# Use a color preset
tf --style dracula README.md

# Use a syntax highlighting theme
tf --syntax-style nord file.md

# Disable clipboard integration
tf --no-clipboard document.md

# List available syntax styles
tf --list-syntax-styles
```

### CLI Options

```
usage: tf [-h] [-w N] [-c PATH] [--style {default,dracula,nord,gruvbox}]
          [--syntax-style NAME] [--list-syntax-styles] [--no-clipboard]
          [--no-hyperlinks] [--no-pretty] [-V]
          [file]

options:
  -h, --help            show this help message and exit
  -w, --width N         Terminal width (default: auto-detect)
  -c, --config PATH     Path to config file
  --style PRESET        Color style preset (default, dracula, nord, gruvbox)
  --syntax-style NAME   Pygments syntax highlighting style
  --list-syntax-styles  List available syntax highlighting styles
  --no-clipboard        Disable OSC 52 clipboard for code blocks
  --no-hyperlinks       Disable OSC 8 hyperlinks
  --no-pretty           Disable pretty code block borders
  -V, --version         show program's version number and exit
```

### Programmatic Usage

```python
from termflow import Parser, Renderer, render_markdown

# Quick rendering to stdout
render_markdown("# Hello World!")

# Render to a file or buffer
from io import StringIO

output = StringIO()
render_markdown("# Hello\n\nThis is **bold** text.", output=output)
print(output.getvalue())
```

### Streaming Mode

For real-time rendering of streaming content (e.g., LLM responses):

```python
from termflow import Parser, Renderer
import sys

# Create parser and renderer
parser = Parser()
renderer = Renderer(output=sys.stdout, width=80)

# Process markdown line by line as it streams in
for line in markdown_stream:
    events = parser.parse_line(line)
    renderer.render_all(events)

# Finalize to close any open blocks
renderer.render_all(parser.finalize())
```

### Custom Styling

```python
from termflow import Renderer, RenderStyle, RenderFeatures
from io import StringIO

# Use a preset style
style = RenderStyle.dracula()  # or .nord(), .gruvbox()

# Or create a custom style
style = RenderStyle(
    bright="#87ceeb",   # Main accent color
    head="#98fb98",     # Heading color
    symbol="#dda0dd",   # Bullets, borders
    link="#87cefa",     # Link color
)

# Configure features
features = RenderFeatures(
    clipboard=True,     # OSC 52 clipboard support
    hyperlinks=True,    # OSC 8 clickable links
    pretty_pad=True,    # Pretty code block borders
)

# Create renderer with custom config
output = StringIO()
renderer = Renderer(
    output=output,
    width=100,
    style=style,
    features=features,
)
```

## 🔧 Configuration

Create a config file at `~/.config/termflow/config.toml`:

```toml
# Terminal width (null = auto-detect)
width = null
max_width = 120

# Pygments syntax highlighting style
syntax_style = "monokai"

# Color scheme
[style]
bright = "#87ceeb"    # Main accent color
head = "#98fb98"      # Heading color  
symbol = "#dda0dd"    # Bullets, table borders, code block borders
grey = "#808080"      # Muted text
dark = "#404040"      # Dark accents
mid = "#a0a0a0"       # Medium text
light = "#d0d0d0"     # Light accents
link = "#87cefa"      # Hyperlink color
error = "#ff6b6b"     # Error messages

# Feature toggles
[features]
clipboard = true      # OSC 52 clipboard for code blocks
hyperlinks = true     # OSC 8 clickable links
pretty_pad = true     # Pretty unicode borders on code blocks
```

You can also set the config path via environment variable:

```bash
export TERMFLOW_CONFIG=/path/to/config.toml
```

## 🎨 Style Presets

termflow includes several built-in color presets:

| Preset | Description |
|--------|-------------|
| `default` | Soft pastel colors |
| `dracula` | Purple-tinted dark theme |
| `nord` | Arctic, bluish color palette |
| `gruvbox` | Retro, earthy colors |

Use them via CLI:

```bash
tf --style dracula README.md
```

Or programmatically:

```python
from termflow import RenderStyle

style = RenderStyle.dracula()
style = RenderStyle.nord()
style = RenderStyle.gruvbox()
```

## 💭 Think Block Support

termflow has special support for `<think>` blocks, commonly used in LLM
chain-of-thought prompting:

```markdown
<think>
Let me reason through this step by step...
1. First, I'll analyze the problem
2. Then, I'll formulate a solution
</think>

Here's my answer based on my reasoning above.
```

Think blocks are rendered with a distinct style to visually separate the
model's reasoning from its final response.

## 🦀 Origin

termflow is a Python port of [streamdown-rs](https://github.com/streamdown-rs/streamdown),
a high-performance streaming markdown renderer written in Rust. This project brings
the same streaming rendering capabilities to the Python ecosystem with a clean,
Pythonic API.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```bash
# Clone and install for development
git clone https://github.com/your-username/termflow.git
cd termflow
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check .
ruff format .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ for the terminal

</div>
