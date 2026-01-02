# Welcome to termflow! 🌊

This document tests **every feature** of termflow, a *streaming markdown renderer* for modern terminals. Let's dive in!

---

## Heading Levels

termflow supports all six heading levels, each styled distinctly.

### Third Level: Features

#### Fourth Level: Implementation Details

##### Fifth Level: Technical Notes

###### Sixth Level: Footnotes & Credits

---

## Inline Formatting Showcase

Here's everything you can do with inline styles:

- **Bold text** makes things stand out
- *Italic text* adds emphasis
- ***Bold and italic*** when you really mean it
- `inline code` for technical terms
- ~~Strikethrough~~ for deprecated stuff
- Links like [termflow on GitHub](https://github.com/termflow/termflow) are clickable!
- Even images work: ![Python Logo](https://python.org/logo.png)

You can also mix them: **This is bold with `code` inside** and *italic with ~~strikethrough~~* for maximum expressiveness.

---

## Code Blocks with Syntax Highlighting

One of termflow's best features is **Pygments-powered syntax highlighting**!

### Python Example

```python
from termflow import Parser, Renderer, render_markdown
import sys

def stream_markdown(lines: list[str]) -> None:
    """Stream markdown content line by line."""
    parser = Parser()
    renderer = Renderer(output=sys.stdout, width=80)
    
    for line in lines:
        events = parser.parse_line(line)
        renderer.render_all(events)
    
    # Don't forget to finalize!
    renderer.render_all(parser.finalize())

if __name__ == "__main__":
    stream_markdown(["# Hello", "", "World!"])
```

### JavaScript Example

```javascript
// Streaming markdown in Node.js style
const readline = require('readline');

async function* streamLines(input) {
    const rl = readline.createInterface({ input });
    for await (const line of rl) {
        yield line;
    }
}

const processMarkdown = async (stream) => {
    for await (const line of streamLines(stream)) {
        console.log(`Processing: ${line}`);
    }
};

processMarkdown(process.stdin);
```

### Rust Example

```rust
use std::io::{self, BufRead};

/// A simple streaming markdown concept in Rust
fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let reader = stdin.lock();
    
    for line in reader.lines() {
        let line = line?;
        // In real streamdown-rs, this would parse & render
        println!("📝 {}", line);
    }
    
    Ok(())
}
```

### Plain Code Block (no language)

```
This is a plain code block without syntax highlighting.
It's useful for:
  - ASCII art
  - Plain text output
  - Configuration snippets
  
  ╭──────────────────╮
  │  termflow 🌊     │
  │  v0.1.0          │
  ╰──────────────────╯
```

---

## Lists

### Bullet Lists (Nested 3 Levels Deep)

- **Level 1**: Core Features
  - *Level 2*: Streaming Parser
    - Level 3: Line-by-line processing
    - Level 3: Event-based architecture
    - Level 3: Stateful parsing
  - *Level 2*: Rich Renderer
    - Level 3: ANSI color output
    - Level 3: Unicode box drawing
    - Level 3: Configurable styles
- **Level 1**: Terminal Integration
  - *Level 2*: OSC Sequences
    - Level 3: OSC 52 clipboard
    - Level 3: OSC 8 hyperlinks
  - *Level 2*: Width Detection
    - Level 3: Auto-detect terminal size
    - Level 3: Responsive wrapping
- **Level 1**: Configuration
  - *Level 2*: TOML config files
  - *Level 2*: Environment variables
  - *Level 2*: CLI flags

### Ordered Lists (Nested)

1. **Install termflow**
   1. Using pip: `pip install termflow-md`
   2. Using uvx: `uvx termflow`
   3. From source:
      1. Clone the repository
      2. Run `pip install -e ".[dev]"`
      3. Verify with `tf --version`
2. **Configure your setup**
   1. Create `~/.config/termflow/config.toml`
   2. Choose a color preset
   3. Customize as needed
3. **Start rendering!**
   1. Pipe content: `cat README.md | tf`
   2. Direct file: `tf document.md`
   3. From LLM: `llm query | tf`

### Task Lists (Checkboxes)

Track your progress with GitHub-style task lists:

- [x] Install termflow
- [x] Read the documentation
- [x] Try the CLI with `tf README.md`
- [ ] Configure custom colors
- [ ] Set up config file at `~/.config/termflow/config.toml`
- [ ] Integrate with your LLM workflow

Nested task lists work too:

- [x] **Phase 1: Setup**
  - [x] Clone repository
  - [x] Install dependencies
  - [x] Run tests
- [ ] **Phase 2: Customization**
  - [x] Choose a color preset
  - [ ] Create custom style
  - [ ] Configure keybindings
- [ ] **Phase 3: Integration**
  - [ ] Add to shell aliases
  - [ ] Connect to LLM API
  - [ ] Profit! 🚀

---

## Tables

GitHub-Flavored Markdown tables are fully supported:

| Feature | Status | Notes |
|---------|--------|-------|
| Headings (H1-H6) | ✅ Complete | All levels styled |
| Inline Formatting | ✅ Complete | Bold, italic, code, etc. |
| Code Blocks | ✅ Complete | 100+ languages via Pygments |
| Tables | ✅ Complete | GFM-style with alignment |
| Lists | ✅ Complete | Nested bullet & ordered |
| Think Blocks | ✅ Complete | LLM chain-of-thought |
| Streaming | ✅ Complete | Line-by-line rendering |

Here's another table with different content:

| Preset | Primary Color | Best For |
|--------|---------------|----------|
| default | Sky Blue | General use |
| dracula | Purple | Dark terminals |
| nord | Arctic Blue | Cool aesthetics |
| gruvbox | Warm Yellow | Retro vibes |

---

## Blockquotes

> "The terminal is not just a text interface—it's a canvas for beautiful, 
> efficient communication between humans and machines."
> 
> — *The termflow Philosophy*

Nested blockquotes work too:

> **User Question:**
> How do I render streaming markdown?
>
> > **termflow Answer:**
> > Use the `Parser` and `Renderer` classes together!
> > 
> > > **Pro Tip:** Don't forget to call `parser.finalize()` 
> > > at the end to close any open blocks.

---

## Think Blocks (LLM Chain-of-Thought)

This is a special feature for rendering LLM output with visible reasoning:

<think>
Let me analyze how to best demonstrate termflow's capabilities...

1. First, I should consider what makes a good markdown renderer:
   - Speed: Must handle streaming input
   - Beauty: ANSI colors and Unicode
   - Compatibility: Work in most terminals

2. termflow addresses all of these:
   - Event-based parser for streaming
   - Pygments integration for syntax highlighting
   - Graceful fallbacks for limited terminals

3. The streaming architecture is key:
   - Parser emits events as lines come in
   - Renderer immediately outputs formatted text
   - No need to wait for complete input

Conclusion: termflow is well-designed for the LLM era!
</think>

Based on my analysis, **termflow** is the perfect tool for rendering LLM output in real-time. The streaming architecture means you see formatted output immediately, not after the model finishes generating.

---

## Links and References

Here are some useful links:

- [termflow Documentation](https://github.com/termflow/termflow#readme)
- [Pygments Styles](https://pygments.org/styles/)
- [ANSI Escape Codes](https://en.wikipedia.org/wiki/ANSI_escape_code)

And here's an inline link in a sentence: Check out [streamdown-rs](https://github.com/streamdown-rs/streamdown), the Rust project that inspired termflow!

---

## Images

Images are rendered as their alt text with the URL shown:

![termflow logo - a wave emoji representing streaming data](https://termflow.dev/logo.png)

![Screenshot of termflow rendering code](https://termflow.dev/screenshot.png)

---

## Footnotes

termflow supports footnotes[^1] for adding references and citations. They're great for academic writing[^2] or adding context without cluttering the main text[^3].

[^1]: Footnotes appear as superscript numbers in the rendered output.
[^2]: Like citing sources or adding clarifications.
[^3]: termflow renders these with style!

---

## Mixed Content Paragraph

Let's put it all together! In a **production environment**, you might use termflow to render `LLM output` that includes *various markdown elements*. The ~~old way~~ of just printing raw text is over—now we have **beautiful, *styled* output** with `syntax highlighting`, [clickable links](https://example.com), and even support for ***bold italic*** text. Whether you're building a CLI tool[^4] or just want prettier terminal output, termflow has you covered.

[^4]: Command Line Interface tools are termflow's primary use case.

---

## Horizontal Rules

We've been using them throughout, but here are a few more for good measure:

---

They help separate sections visually.

---

And they look great in the terminal!

---

## The End 🎉

If you've made it this far, you've seen **everything termflow can do**:

1. ✅ All heading levels (H1-H6)
2. ✅ Inline formatting (bold, italic, code, strikethrough, links)
3. ✅ Fenced code blocks with syntax highlighting
4. ✅ Nested bullet and ordered lists
5. ✅ Task lists with checkboxes
6. ✅ GFM tables
7. ✅ Blockquotes (including nested)
8. ✅ Think blocks for LLM output
9. ✅ Horizontal rules
10. ✅ Images and footnotes
11. ✅ Mixed content paragraphs

**Happy rendering!** 🌊

---

*Made with ❤️ by the termflow contributors*
