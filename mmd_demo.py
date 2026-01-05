#!/usr/bin/env python3
"""🎨 Mermaid Diagram Demo for termflow.

Watch flowcharts render in real-time!

Usage:
    python mmd_demo.py
    uv run python mmd_demo.py
"""

from __future__ import annotations

import os
import sys
import time

# =============================================================================
# ANSI Escape Codes
# =============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"

CLEAR = "\033[2J\033[H"

# =============================================================================
# Banner
# =============================================================================

BANNER = rf"""
{BRIGHT_CYAN}{BOLD}
   ╔╦╗┌─┐┬─┐┌┬┐┌─┐┬  ┌─┐┬ ┬  ┌┬┐┌─┐┬─┐┌┬┐┌─┐┬┌┬┐
    ║ ├┤ ├┬┘│││├┤ │  │ ││││  │││├┤ ├┬┘│││├─┤│ ││
    ╩ └─┘┴└─┴ ┴└  ┴─┘└─┘└┴┘  ┴ ┴└─┘┴└─┴ ┴┴ ┴┴─┴┘
{RESET}
{BRIGHT_GREEN}              📊 Flowcharts in your terminal!{RESET}
"""

# =============================================================================
# Demo Content (4 demos only)
# =============================================================================

DEMOS = [
    {
        "title": "🚀 Simple Flowchart",
        "subtitle": "The basics - a linear process flow",
        "color": BRIGHT_GREEN,
        "markdown": """# Simple Flowchart

A basic left-to-right process:

```mermaid
graph LR
    A[Start] --> B[Process Data]
    B --> C[Validate]
    C --> D[Complete]
```

*Simple and clean!*
""",
    },
    {
        "title": "🌳 Decision Tree",
        "subtitle": "Branching logic with conditionals",
        "color": BRIGHT_YELLOW,
        "markdown": """# Decision Tree

User authentication flow:

```mermaid
graph TD
    A[User Request] --> B{Authenticated?}
    B -->|Yes| C[Load Profile]
    B -->|No| D[Show Login]
    C --> E{Has Permission?}
    E -->|Yes| F[Process Request]
    E -->|No| G[Access Denied]
    D --> H[Redirect]
```

*Decisions, decisions...*
""",
    },
    {
        "title": "🏗️ Software Architecture",
        "subtitle": "Microservices overview",
        "color": BRIGHT_MAGENTA,
        "markdown": """# Microservices Architecture

```mermaid
graph TD
    Client[Browser] --> Gateway{API Gateway}
    Gateway --> Auth[Auth Service]
    Gateway --> Users[User Service]
    Gateway --> Orders[Order Service]
    Auth --> DB1[(Auth DB)]
    Users --> DB2[(User DB)]
    Orders --> DB3[(Order DB)]
    Orders --> Queue((Message Queue))
    Queue --> Email[Email Service]
```

*Scalable and distributed!*
""",
    },
    {
        "title": "🔄 CI/CD Pipeline",
        "subtitle": "From code to production",
        "color": BRIGHT_BLUE,
        "markdown": """# CI/CD Pipeline

```mermaid
graph LR
    A[Push Code] --> B[Run Tests]
    B --> C{Tests Pass?}
    C -->|Yes| D[Build Image]
    C -->|No| E[Notify Dev]
    D --> F[Deploy Staging]
    F --> G{QA Approved?}
    G -->|Yes| H[Deploy Prod]
    G -->|No| E
```

*Ship it!* 🚢
""",
    },
]

# =============================================================================
# Utilities
# =============================================================================


def get_terminal_width() -> int:
    """Get terminal width."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


# Demo uses EXTRA LARGE diagrams for maximum impact!
DEMO_MERMAID_CONFIG = None  # Lazy-loaded


def get_demo_mermaid_config():
    """Get the demo mermaid config."""
    global DEMO_MERMAID_CONFIG
    if DEMO_MERMAID_CONFIG is None:
        from termflow.render.mermaid import RenderConfig

        DEMO_MERMAID_CONFIG = RenderConfig(
            size_multiplier=1.2,  # Just 20% bigger than default
            stroke_width=4,       # Default thickness
            font_size=24,         # Readable but not overwhelming
        )
    return DEMO_MERMAID_CONFIG


def stream_markdown(markdown: str, line_delay: float = 0.06) -> None:
    """Stream markdown through termflow renderer."""
    from termflow import Parser, Renderer

    parser = Parser()
    width = min(get_terminal_width(), 100)

    renderer = Renderer(
        output=sys.stdout,
        width=width,
        mermaid_render_config=get_demo_mermaid_config(),
    )

    for line in markdown.strip().split("\n"):
        time.sleep(line_delay)
        events = parser.parse_line(line)
        renderer.render_all(events)

    renderer.render_all(parser.finalize())


def run_demo_item(demo: dict, index: int, total: int) -> None:
    """Run a single demo item."""
    color = demo["color"]

    print()
    print(f"{color}{'━' * 60}{RESET}")
    print(f"{color}{BOLD}  Demo {index}/{total}: {demo['title']}{RESET}")
    print(f"{DIM}  {demo['subtitle']}{RESET}")
    print(f"{color}{'━' * 60}{RESET}")
    print()

    stream_markdown(demo["markdown"])

    print(f"\n{BRIGHT_GREEN}✓ Rendered!{RESET}")


# =============================================================================
# Main
# =============================================================================


def run_demo() -> None:
    """Run the full demo - fully automated."""
    try:
        # Quick intro
        print(CLEAR)
        print(BANNER)
        print(f"{DIM}  Rendering {len(DEMOS)} diagrams...{RESET}")
        time.sleep(1)

        # Run each demo
        for i, demo in enumerate(DEMOS, 1):
            print(CLEAR)
            run_demo_item(demo, i, len(DEMOS))

            # Pause between demos (except after last one)
            if i < len(DEMOS):
                time.sleep(3)

        # Outro
        time.sleep(1)
        print()
        print(f"{BRIGHT_CYAN}{'═' * 60}{RESET}")
        print(f"{BRIGHT_WHITE}{BOLD}  🎉 Demo Complete!{RESET}")
        print(f"{BRIGHT_CYAN}{'═' * 60}{RESET}")
        print()
        print(f"  {BRIGHT_GREEN}tf examples/mermaid.md{RESET}  {DIM}# Try it yourself{RESET}")
        print(f"  {BRIGHT_GREEN}pip install termflow-md{RESET} {DIM}# Install termflow{RESET}")
        print()

    except KeyboardInterrupt:
        print(f"\n{BRIGHT_YELLOW}Interrupted. Bye! 👋{RESET}\n")
        sys.exit(0)


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="🎨 Mermaid Diagram Demo for termflow",
    )
    parser.add_argument(
        "--single",
        type=int,
        metavar="N",
        help=f"Run only demo N (1-{len(DEMOS)})",
    )

    args = parser.parse_args()

    if args.single:
        if 1 <= args.single <= len(DEMOS):
            print(CLEAR)
            run_demo_item(DEMOS[args.single - 1], args.single, len(DEMOS))
            print()
        else:
            print(f"Error: Demo number must be 1-{len(DEMOS)}")
            sys.exit(1)
    else:
        run_demo()


if __name__ == "__main__":
    main()
