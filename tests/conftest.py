"""Pytest configuration and shared fixtures for termflow tests."""

from io import StringIO

import pytest

from termflow.parser import Parser
from termflow.render import Renderer, RenderFeatures, RenderStyle
from termflow.syntax import Highlighter


@pytest.fixture
def parser():
    """Create a fresh Parser instance."""
    return Parser()


@pytest.fixture
def output():
    """Create a StringIO buffer for capturing output."""
    return StringIO()


@pytest.fixture
def renderer(output):
    """Create a Renderer with default settings."""
    return Renderer(output=output, width=80)


@pytest.fixture
def renderer_narrow(output):
    """Create a Renderer with narrow width."""
    return Renderer(output=output, width=40)


@pytest.fixture
def renderer_wide(output):
    """Create a Renderer with wide width."""
    return Renderer(output=output, width=120)


@pytest.fixture
def highlighter():
    """Create a Highlighter instance."""
    return Highlighter()


@pytest.fixture
def style_default():
    """Default RenderStyle."""
    return RenderStyle()


@pytest.fixture
def style_dracula():
    """Dracula RenderStyle."""
    return RenderStyle.dracula()


@pytest.fixture
def style_nord():
    """Nord RenderStyle."""
    return RenderStyle.nord()


@pytest.fixture
def features_all_enabled():
    """RenderFeatures with all features enabled."""
    return RenderFeatures(
        clipboard=True,
        hyperlinks=True,
        pretty_pad=True,
    )


@pytest.fixture
def features_all_disabled():
    """RenderFeatures with all features disabled."""
    return RenderFeatures(
        clipboard=False,
        hyperlinks=False,
        pretty_pad=False,
    )


# Sample markdown documents for testing
@pytest.fixture
def sample_markdown_simple():
    """Simple markdown document."""
    return """# Hello World

This is a paragraph.
"""


@pytest.fixture
def sample_markdown_complex():
    """Complex markdown document with many features."""
    return """# Main Title

This is a paragraph with **bold**, *italic*, and `code`.

## Code Example

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

## Lists

- Bullet item 1
- Bullet item 2
  - Nested item
- Bullet item 3

1. Ordered item 1
2. Ordered item 2

## Table

| Name | Age | City |
|------|-----|------|
| Alice | 30 | NYC |
| Bob | 25 | LA |

## Quote

> This is a blockquote.
> It can span multiple lines.

---

[Link to example](https://example.com)

The end.
"""


@pytest.fixture
def sample_markdown_code():
    """Markdown with code blocks in multiple languages."""
    return """## Python

```python
print("Hello, World!")
```

## JavaScript

```javascript
console.log("Hello, World!");
```

## Rust

```rust
fn main() {
    println!("Hello, World!");
}
```
"""
