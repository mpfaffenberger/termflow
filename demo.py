#!/usr/bin/env python3
"""
termflow demo - Watch markdown render in real-time! 🌊

This simulates LLM-style streaming output to showcase termflow's
streaming rendering capabilities.

Usage:
    python demo.py [file.md]
    python demo.py              # defaults to test_document.md
    python demo.py --fast       # 2x speed
    python demo.py --slow       # 0.5x speed
    python demo.py --dim        # dim/faded mode (for thinking blocks)
"""

import random
import sys
import time
from pathlib import Path

from termflow import Parser, Renderer
from termflow.ansi import BOLD_OFF, BOLD_ON, DIM_OFF, DIM_ON, RESET, fg_color

# =============================================================================
# Colors
# =============================================================================

CYAN = fg_color("#87ceeb")
GREEN = fg_color("#50fa7b")
PURPLE = fg_color("#bd93f9")
YELLOW = fg_color("#f1fa8c")
PINK = fg_color("#ff79c6")
ORANGE = fg_color("#ffb86c")
GREY = fg_color("#6272a4")

# =============================================================================
# ASCII Art
# =============================================================================

LOGO = f"""
{CYAN}{BOLD_ON}
  ╭─────────────────────────────────────────────────────────╮
  │                                                         │
  │   {PURPLE}███████████████████████████████████████████████████{CYAN}   │
  │   {PURPLE}██{RESET}   _                       __ _                {PURPLE}██{CYAN}   │
  │   {PURPLE}██{RESET}  | |_ ___ _ __ _ __ ___  / _| | _____      __ {PURPLE}██{CYAN}   │
  │   {PURPLE}██{RESET}  | __/ _ \\ '__| '_ ` _ \\| |_| |/ _ \\ \\ /\\ / / {PURPLE}██{CYAN}   │
  │   {PURPLE}██{RESET}  | ||  __/ |  | | | | | |  _| | (_) \\ V  V /  {PURPLE}██{CYAN}   │
  │   {PURPLE}██{RESET}   \\__\\___|_|  |_| |_| |_|_| |_|\\___/ \\_/\\_/   {PURPLE}██{CYAN}   │
  │   {PURPLE}██{RESET}                                     {PINK}🌊 v0.1.0{RESET} {PURPLE}██{CYAN}   │
  │   {PURPLE}███████████████████████████████████████████████████{CYAN}   │
  │                                                         │
  ╰─────────────────────────────────────────────────────────╯
{BOLD_OFF}{RESET}
"""

WAVE_FRAMES = ["🌊", "💧", "🌊", "✨"]


# =============================================================================
# Streaming Demo
# =============================================================================


def stream_file(
    filepath: Path,
    chars_per_chunk: tuple[int, int] = (4, 6),
    chunks_per_second: float = 15.0,
    dim: bool = False,
) -> tuple[int, int]:
    """Stream a file through termflow with realistic LLM-like timing.

    Args:
        filepath: Path to markdown file to stream.
        chars_per_chunk: Min/max characters per chunk (randomized).
        chunks_per_second: How many chunks to emit per second.
        dim: If True, render in dim/faded mode.

    Returns:
        Tuple of (total_chars, total_lines) processed.
    """
    content = filepath.read_text()
    delay = 1.0 / chunks_per_second

    parser = Parser()
    renderer = Renderer(dim=dim)

    # Buffer for accumulating characters until we hit a newline
    line_buffer = ""
    total_chars = 0
    total_lines = 0

    i = 0
    while i < len(content):
        # Random chunk size for realism
        chunk_size = random.randint(*chars_per_chunk)
        chunk = content[i : i + chunk_size]
        i += chunk_size
        total_chars += len(chunk)

        # Add chunk to buffer
        line_buffer += chunk

        # Process complete lines
        while "\n" in line_buffer:
            line, line_buffer = line_buffer.split("\n", 1)
            events = parser.parse_line(line)
            renderer.render_all(events)
            total_lines += 1

        # Small delay between chunks
        time.sleep(delay)

    # Process any remaining content in buffer
    if line_buffer:
        events = parser.parse_line(line_buffer)
        renderer.render_all(events)
        total_lines += 1

    # Finalize to close any open blocks
    renderer.render_all(parser.finalize())

    return total_chars, total_lines


def print_intro(filepath: Path, speed: float, dim: bool = False) -> None:
    """Print a cute intro message."""
    print(LOGO)

    speed_label = {
        0.5: f"{ORANGE}slow 🐢{RESET}",
        1.0: f"{GREEN}normal 🐕{RESET}",
        2.0: f"{PINK}fast 🚀{RESET}",
    }.get(speed, f"{YELLOW}{speed}x{RESET}")

    dim_label = f"{GREY}dim 🌙{RESET}" if dim else f"{GREEN}normal{RESET}"

    print(f"  {GREY}──────────────────────────────────────────────{RESET}")
    print()
    print(f"  {BOLD_ON}Streaming Demo{BOLD_OFF} {DIM_ON}- Simulating LLM output{DIM_OFF}")
    print()
    print(f"  {CYAN}▸{RESET} File: {YELLOW}{filepath.name}{RESET}")
    print(f"  {CYAN}▸{RESET} Speed: {speed_label}")
    print(f"  {CYAN}▸{RESET} Style: {dim_label}")
    print(f"  {CYAN}▸{RESET} Press {PINK}Ctrl+C{RESET} to stop")
    print()
    print(f"  {GREY}──────────────────────────────────────────────{RESET}")
    print()

    # Countdown
    for i in range(3, 0, -1):
        frame = WAVE_FRAMES[i % len(WAVE_FRAMES)]
        sys.stdout.write(f"\r  {frame} Starting in {PURPLE}{i}{RESET}...   ")
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write(f"\r  {GREEN}✓{RESET} Let's go!            \n")
    print()
    time.sleep(0.3)


def print_outro(elapsed: float, total_chars: int, total_lines: int) -> None:
    """Print a cute outro with stats."""
    print()
    print()
    print(f"  {GREY}──────────────────────────────────────────────{RESET}")
    print()
    print(f"  {GREEN}{BOLD_ON}✓ Demo complete!{BOLD_OFF}{RESET} 🎉")
    print()
    print(f"  {CYAN}▸{RESET} Time elapsed: {YELLOW}{elapsed:.2f}s{RESET}")
    print(f"  {CYAN}▸{RESET} Characters: {PURPLE}{total_chars:,}{RESET}")
    print(f"  {CYAN}▸{RESET} Lines: {PURPLE}{total_lines:,}{RESET}")
    print(f"  {CYAN}▸{RESET} Speed: {ORANGE}{total_chars / elapsed:.0f}{RESET} chars/sec")
    print()
    print(f"  {DIM_ON}Try it yourself:{DIM_OFF}")
    print(f"  {GREY}${RESET} echo '# Hello **World**' | tf")
    print(f"  {GREY}${RESET} tf README.md")
    print(f"  {GREY}${RESET} cat doc.md | tf --style dracula")
    print()
    print(f"  {GREY}──────────────────────────────────────────────{RESET}")
    print()


def print_interrupted(elapsed: float, total_chars: int) -> None:
    """Print message when interrupted."""
    print()
    print()
    print(f"  {YELLOW}{BOLD_ON}⚠ Interrupted{BOLD_OFF}{RESET}")
    print(f"  {DIM_ON}Streamed {total_chars:,} chars in {elapsed:.2f}s{DIM_OFF}")
    print()


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Run the streaming demo."""
    # Parse arguments
    args = sys.argv[1:]
    speed = 1.0
    dim = False
    filepath = Path("test_document.md")

    for arg in args:
        if arg == "--fast":
            speed = 2.0
        elif arg == "--slow":
            speed = 0.5
        elif arg == "--dim":
            dim = True
        elif arg == "--help" or arg == "-h":
            print(__doc__)
            return 0
        elif not arg.startswith("--"):
            filepath = Path(arg)

    # Validate file exists
    if not filepath.exists():
        print(f"{PINK}Error:{RESET} File not found: {filepath}")
        return 1

    # Calculate chunks per second based on speed
    base_chunks_per_second = 60
    chunks_per_second = base_chunks_per_second * speed

    # Show intro
    print_intro(filepath, speed, dim)

    # Run the demo
    start_time = time.time()
    total_chars = 0
    total_lines = 0

    try:
        total_chars, total_lines = stream_file(
            filepath,
            chars_per_chunk=(4, 6),
            chunks_per_second=chunks_per_second,
            dim=dim,
        )
        elapsed = time.time() - start_time
        print_outro(elapsed, total_chars, total_lines)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print_interrupted(elapsed, total_chars)
        return 130  # Standard exit code for Ctrl+C

    return 0


if __name__ == "__main__":
    sys.exit(main())
