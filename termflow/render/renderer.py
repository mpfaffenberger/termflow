"""Main terminal renderer for markdown events.

This is the core rendering engine that converts ParseEvents into
beautiful ANSI-formatted terminal output. It handles all event types
and maintains rendering state across events.
"""

from __future__ import annotations

import base64
import os
import sys
from typing import TextIO

from termflow.ansi import (
    BOLD_OFF,
    BOLD_ON,
    DIM_OFF,
    DIM_ON,
    ITALIC_OFF,
    ITALIC_ON,
    RESET,
    STRIKEOUT_OFF,
    STRIKEOUT_ON,
    UNDERLINE_OFF,
    UNDERLINE_ON,
    fg_color,
    make_link,
)
from termflow.parser.events import (
    BlockquoteEndEvent,
    BlockquoteLineEvent,
    BlockquoteStartEvent,
    BoldEvent,
    BoldItalicEvent,
    CodeBlockEndEvent,
    CodeBlockLineEvent,
    CodeBlockStartEvent,
    EmptyLineEvent,
    FootnoteEvent,
    HeadingEvent,
    HorizontalRuleEvent,
    ImageEvent,
    InlineCodeEvent,
    ItalicEvent,
    LinkEvent,
    ListEndEvent,
    ListItemContentEvent,
    ListItemEndEvent,
    ListItemStartEvent,
    NewlineEvent,
    ParagraphEndEvent,
    ParagraphStartEvent,
    ParseEvent,
    StrikeoutEvent,
    TableEndEvent,
    TableHeaderEvent,
    TableRowEvent,
    TableSeparatorEvent,
    TableStartEvent,
    TextEvent,
    ThinkBlockEndEvent,
    ThinkBlockLineEvent,
    ThinkBlockStartEvent,
    UnderlineEvent,
)
from termflow.render.code import render_code_end, render_code_line, render_code_start
from termflow.render.heading import render_heading
from termflow.render.list import get_bullet, render_list_item
from termflow.render.style import RenderFeatures, RenderStyle
from termflow.render.table import (
    TableRenderState,
    render_table_bottom,
    render_table_row,
    render_table_separator,
    render_table_top,
)
from termflow.render.text import text_wrap
from termflow.syntax import Highlighter


class Renderer:
    """Terminal renderer for markdown events.

    Converts ParseEvents into beautiful ANSI-formatted terminal output.

    Attributes:
        output: Output stream for writing
        width: Terminal width in columns
        style: Color/style configuration
        features: Feature flags
        highlighter: Syntax highlighter instance

    Example:
        >>> renderer = Renderer(width=80)
        >>> for event in parser.parse_line(line):
        ...     renderer.render(event)

        >>> # Or render multiple events
        >>> renderer.render_all(events)
    """

    def __init__(
        self,
        output: TextIO | None = None,
        width: int | None = None,
        style: RenderStyle | None = None,
        features: RenderFeatures | None = None,
        highlighter: Highlighter | None = None,
    ) -> None:
        """Initialize the renderer.

        Args:
            output: Output stream (default: sys.stdout)
            width: Terminal width (default: auto-detect)
            style: Color/style configuration
            features: Feature flags
            highlighter: Syntax highlighter (created if not provided)
        """
        self.output = output or sys.stdout
        self.width = width or self._detect_width()
        self.style = style or RenderStyle()
        self.features = features or RenderFeatures()
        self.highlighter = highlighter or Highlighter()

        # Table state
        self.table_state = TableRenderState()
        self._table_started = False

        # Rendering state
        self._code_language: str | None = None
        self._code_buffer: str = ""
        self._in_blockquote = False
        self._blockquote_depth = 0
        self._list_depth = 0
        self._list_number: int | None = None
        self._list_ordered = False
        self._list_checked: bool | None = None  # For task lists
        self._in_paragraph = False

    @staticmethod
    def _detect_width() -> int:
        """Detect terminal width."""
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80

    def _write(self, text: str) -> None:
        """Write text to output without newline."""
        self.output.write(text)
        self.output.flush()

    def _writeln(self, text: str = "") -> None:
        """Write text to output with newline."""
        self.output.write(text + "\n")
        self.output.flush()

    def _margin(self) -> str:
        """Get current left margin string."""
        if self._in_blockquote:
            fg = fg_color(self.style.symbol)
            return f"{fg}│{RESET} " * self._blockquote_depth
        return ""

    def _current_width(self) -> int:
        """Get current available width accounting for margins."""
        margin_width = self._blockquote_depth * 3 if self._in_blockquote else 0
        return max(20, self.width - margin_width)

    def _format_inline(self, text: str) -> str:
        """Format text with inline formatting, returning ANSI string."""
        from termflow.parser.inline import InlineElement, InlineParser

        parser = InlineParser()
        tokens = parser.parse(text)
        parts: list[str] = []

        for token in tokens:
            t = token.element_type
            if t == InlineElement.TEXT:
                parts.append(token.content)
            elif t == InlineElement.BOLD:
                parts.append(f"{BOLD_ON}{token.content}{BOLD_OFF}")
            elif t == InlineElement.ITALIC:
                parts.append(f"{ITALIC_ON}{token.content}{ITALIC_OFF}")
            elif t == InlineElement.BOLD_ITALIC:
                parts.append(f"{BOLD_ON}{ITALIC_ON}{token.content}{ITALIC_OFF}{BOLD_OFF}")
            elif t == InlineElement.CODE:
                # Render inline code with dim styling (no background)
                parts.append(f"{DIM_ON}`{token.content}`{DIM_OFF}")
            elif t == InlineElement.UNDERLINE:
                parts.append(f"{UNDERLINE_ON}{token.content}{UNDERLINE_OFF}")
            elif t == InlineElement.STRIKEOUT:
                parts.append(f"{STRIKEOUT_ON}{token.content}{STRIKEOUT_OFF}")
            elif t == InlineElement.LINK:
                link_fg = fg_color(self.style.link)
                grey = fg_color(self.style.grey)
                link_text = f"{link_fg}{token.content}{RESET}"
                if self.features.hyperlinks and token.url:
                    parts.append(make_link(token.url, link_text))
                else:
                    parts.append(f"{UNDERLINE_ON}{link_text}{UNDERLINE_OFF}")
                if token.url:
                    parts.append(f" {grey}({token.url}){RESET}")
            elif t == InlineElement.IMAGE:
                symbol_fg = fg_color(self.style.symbol)
                parts.append(f"{symbol_fg}[🖼 {token.content}]{RESET}")
            elif t == InlineElement.FOOTNOTE:
                symbol_fg = fg_color(self.style.symbol)
                parts.append(f"{symbol_fg}[{token.content}]{RESET}")
            else:
                parts.append(token.content)

        return "".join(parts)

    def _render_inline_text(self, text: str) -> None:
        """Render text with inline formatting (bold, italic, etc.)."""
        self._write(self._format_inline(text))

    def render(self, event: ParseEvent) -> None:
        """Render a single parse event.

        Args:
            event: The parse event to render.
        """
        # === Inline Events ===
        if isinstance(event, TextEvent):
            # Parse and render inline formatting
            self._render_inline_text(event.text)

        elif isinstance(event, InlineCodeEvent):
            # Render inline code with dim styling (no background)
            self._write(f"{DIM_ON}`{event.code}`{DIM_OFF}")

        elif isinstance(event, BoldEvent):
            self._write(f"{BOLD_ON}{event.text}{BOLD_OFF}")

        elif isinstance(event, ItalicEvent):
            self._write(f"{ITALIC_ON}{event.text}{ITALIC_OFF}")

        elif isinstance(event, BoldItalicEvent):
            self._write(f"{BOLD_ON}{ITALIC_ON}{event.text}{ITALIC_OFF}{BOLD_OFF}")

        elif isinstance(event, UnderlineEvent):
            self._write(f"{UNDERLINE_ON}{event.text}{UNDERLINE_OFF}")

        elif isinstance(event, StrikeoutEvent):
            self._write(f"{STRIKEOUT_ON}{event.text}{STRIKEOUT_OFF}")

        elif isinstance(event, LinkEvent):
            link_fg = fg_color(self.style.link)
            grey = fg_color(self.style.grey)

            if self.features.hyperlinks:
                self._write(make_link(event.url, f"{link_fg}{event.text}{RESET}"))
            else:
                self._write(f"{UNDERLINE_ON}{link_fg}{event.text}{RESET}{UNDERLINE_OFF}")

            # Show URL in grey
            self._write(f" {grey}({event.url}){RESET}")

        elif isinstance(event, ImageEvent):
            fg = fg_color(self.style.symbol)
            self._write(f"{fg}[🖼 {event.alt}]{RESET}")
            if event.url:
                grey = fg_color(self.style.grey)
                self._write(f" {grey}({event.url}){RESET}")

        elif isinstance(event, FootnoteEvent):
            fg = fg_color(self.style.symbol)
            # Use superscript for footnotes
            from termflow.ansi import number_to_superscript

            try:
                num = int(event.number)
                super_num = number_to_superscript(num)
            except ValueError:
                super_num = f"[{event.number}]"
            self._write(f"{fg}{super_num}{RESET}")

        # === Heading Events ===
        elif isinstance(event, HeadingEvent):
            margin = self._margin()
            width = self._current_width()
            for line in render_heading(event.level, event.content, width, margin, self.style):
                self._writeln(line)

        # === Code Block Events ===
        elif isinstance(event, CodeBlockStartEvent):
            self._code_language = event.language
            self._code_buffer = ""

            margin = self._margin()
            width = self._current_width()
            for line in render_code_start(
                event.language, width, margin, self.style, self.features.pretty_pad
            ):
                self._writeln(line)

        elif isinstance(event, CodeBlockLineEvent):
            # Accumulate code for clipboard
            if self._code_buffer:
                self._code_buffer += "\n"
            self._code_buffer += event.line

            # Highlight and render
            highlighted = self.highlighter.highlight_line(event.line, self._code_language or "text")
            margin = self._margin()
            width = self._current_width()
            line = render_code_line(
                event.line, highlighted, width, margin, self.style, self.features.pretty_pad
            )
            self._writeln(line)

        elif isinstance(event, CodeBlockEndEvent):
            margin = self._margin()
            width = self._current_width()
            for line in render_code_end(width, margin, self.style, self.features.pretty_pad):
                self._writeln(line)

            # Clipboard integration (OSC 52)
            if self.features.clipboard and self._code_buffer:
                self._copy_to_clipboard(self._code_buffer)

            self._code_language = None
            self._code_buffer = ""

        # === List Events ===
        elif isinstance(event, ListItemStartEvent):
            self._list_depth = event.indent
            self._list_ordered = event.ordered
            self._list_number = event.number
            self._list_checked = event.checked

        elif isinstance(event, ListItemContentEvent):
            margin = self._margin()
            width = self._current_width()

            # Determine bullet
            if self._list_checked is not None:
                # Task list item - use checkbox
                bullet = "☑" if self._list_checked else "☐"
            elif self._list_ordered and self._list_number is not None:
                bullet = f"{self._list_number}."
            else:
                bullet = get_bullet(self._list_depth)

            # Format inline content
            formatted_content = self._format_inline(event.text)

            for line in render_list_item(
                self._list_depth,
                bullet,
                formatted_content,
                width,
                margin,
                self.style,
                is_ordered=self._list_ordered,
                number=self._list_number,
            ):
                self._writeln(line)

        elif isinstance(event, ListItemEndEvent):
            pass  # No action needed

        elif isinstance(event, ListEndEvent):
            self._list_depth = 0
            self._list_ordered = False
            self._list_number = None
            self._list_checked = None

        # === Table Events ===
        elif isinstance(event, TableStartEvent):
            self.table_state.reset()
            self._table_started = True

        elif isinstance(event, TableHeaderEvent):
            if not self._table_started:
                self.table_state.reset()
                self._table_started = True

            margin = self._margin()
            width = self._current_width()

            # First update widths
            self.table_state.update_widths(event.cells)

            # Render top border
            self._writeln(render_table_top(self.table_state, margin, self.style))

            # Render header row
            for line in render_table_row(
                list(event.cells), self.table_state, width, margin, self.style, is_header=True
            ):
                self._writeln(line)

        elif isinstance(event, TableSeparatorEvent):
            margin = self._margin()

            # Update alignments
            if event.alignments:
                self.table_state.set_alignments(list(event.alignments))

            self._writeln(render_table_separator(self.table_state, margin, self.style))
            self.table_state.end_header()

        elif isinstance(event, TableRowEvent):
            margin = self._margin()
            width = self._current_width()

            for line in render_table_row(
                list(event.cells), self.table_state, width, margin, self.style, is_header=False
            ):
                self._writeln(line)

        elif isinstance(event, TableEndEvent):
            if self._table_started:
                margin = self._margin()
                self._writeln(render_table_bottom(self.table_state, margin, self.style))

            self.table_state.reset()
            self._table_started = False

        # === Blockquote Events ===
        elif isinstance(event, BlockquoteStartEvent):
            self._in_blockquote = True
            self._blockquote_depth = event.depth

        elif isinstance(event, BlockquoteLineEvent):
            margin = self._margin()
            width = self._current_width()

            # Format inline content
            formatted_text = self._format_inline(event.text)

            for line in text_wrap(formatted_text, width, 0, margin, margin):
                self._writeln(line)

        elif isinstance(event, BlockquoteEndEvent):
            self._in_blockquote = False
            self._blockquote_depth = 0

        # === Think Block Events ===
        elif isinstance(event, ThinkBlockStartEvent):
            fg = fg_color(self.style.grey)
            dim = DIM_ON
            self._writeln(f"{dim}{fg}┌── thinking ──{RESET}")
            self._in_blockquote = True
            self._blockquote_depth = 1

        elif isinstance(event, ThinkBlockLineEvent):
            fg = fg_color(self.style.grey)
            dim = DIM_ON
            formatted_text = self._format_inline(event.text)
            self._writeln(f"{dim}{fg}│{RESET} {dim}{formatted_text}{DIM_OFF}")

        elif isinstance(event, ThinkBlockEndEvent):
            fg = fg_color(self.style.grey)
            dim = DIM_ON
            self._writeln(f"{dim}{fg}└──{RESET}")
            self._in_blockquote = False
            self._blockquote_depth = 0

        # === Other Block Events ===
        elif isinstance(event, HorizontalRuleEvent):
            fg = fg_color(self.style.grey)
            margin = self._margin()
            rule = "─" * self._current_width()
            self._writeln(f"{margin}{fg}{rule}{RESET}")

        elif isinstance(event, EmptyLineEvent | NewlineEvent):
            self._writeln()

        elif isinstance(event, ParagraphStartEvent):
            self._in_paragraph = True

        elif isinstance(event, ParagraphEndEvent):
            self._in_paragraph = False

    def render_all(self, events: list[ParseEvent]) -> None:
        """Render multiple events.

        Args:
            events: List of parse events to render.
        """
        for event in events:
            self.render(event)

    def _copy_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard using OSC 52.

        OSC 52 is supported by many modern terminals including:
        - iTerm2, Kitty, Alacritty, WezTerm, foot
        - tmux (with set-clipboard on)
        - Some versions of xterm

        Args:
            text: Text to copy to clipboard.
        """
        try:
            encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
            # OSC 52 sequence: ESC ] 52 ; c ; BASE64 BEL
            self._write(f"\x1b]52;c;{encoded}\x07")
        except Exception:
            pass  # Silently fail if clipboard doesn't work

    def set_width(self, width: int) -> None:
        """Update the terminal width.

        Args:
            width: New width in columns.
        """
        self.width = width

    def set_style(self, style: RenderStyle) -> None:
        """Update the render style.

        Args:
            style: New style configuration.
        """
        self.style = style

    def reset(self) -> None:
        """Reset renderer state."""
        self.table_state.reset()
        self._table_started = False
        self._code_language = None
        self._code_buffer = ""
        self._in_blockquote = False
        self._blockquote_depth = 0
        self._list_depth = 0
        self._list_number = None
        self._list_ordered = False
        self._list_checked = None
        self._in_paragraph = False


def render_markdown(
    markdown: str,
    width: int | None = None,
    output: TextIO | None = None,
    style: RenderStyle | None = None,
) -> None:
    """Convenience function to render markdown to terminal.

    Args:
        markdown: Markdown text to render.
        width: Terminal width (auto-detect if None).
        output: Output stream (stdout if None).
        style: Render style (default if None).

    Example:
        >>> render_markdown("# Hello\n\nThis is **bold**!")
    """
    from termflow.parser import Parser

    parser = Parser()
    renderer = Renderer(output=output, width=width, style=style)

    events = parser.parse_document(markdown)
    renderer.render_all(events)


def render_streaming(
    lines_iter,
    width: int | None = None,
    output: TextIO | None = None,
    style: RenderStyle | None = None,
) -> None:
    """Render markdown from a streaming source.

    Args:
        lines_iter: Iterator/generator yielding lines.
        width: Terminal width (auto-detect if None).
        output: Output stream (stdout if None).
        style: Render style (default if None).

    Example:
        >>> def stream():
        ...     yield "# Hello"
        ...     yield ""
        ...     yield "World!"
        >>> render_streaming(stream())
    """
    from termflow.parser import Parser

    parser = Parser()
    renderer = Renderer(output=output, width=width, style=style)

    for line in lines_iter:
        events = parser.parse_line(line)
        renderer.render_all(events)

    # Finalize
    final_events = parser.finalize()
    renderer.render_all(final_events)
