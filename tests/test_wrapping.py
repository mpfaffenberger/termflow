"""Word wrapping + live terminal width across the rendering pipeline."""

from __future__ import annotations

import re
from io import StringIO

import pytest

from termflow import Parser
from termflow.ansi import make_link, visible, visible_length, wrap_ansi
from termflow.ansi.utils import OSC8_CLOSE, _get_active_codes, split_ansi
from termflow.render import (
    CODE_CONTINUATION,
    Renderer,
    RenderFeatures,
    render_markdown_lines,
)

WORDS = "Sorting bananas across manifolds requires patience and considerable dexterity"

DOC = f"""# Heading {WORDS}

Paragraph {WORDS}.

- Bullet {WORDS}
  1. Nested {WORDS}

> Quote {WORDS}

<think>
Thinking {WORDS}
</think>
"""


def render(markdown: str, width: int | None = 40, **kwargs) -> str:
    out = StringIO()
    renderer = Renderer(output=out, width=width, **kwargs)
    renderer.render_all(Parser().parse_document(markdown))
    return out.getvalue()


def tokens(rendered: str) -> set[str]:
    return set(visible(rendered).split())


class TestWrapAnsiFixes:
    def test_no_dangling_space_at_wrap_point(self):
        assert wrap_ansi("aaaa bbbb cccc", 9) == ["aaaa bbbb", "cccc"]

    def test_off_code_cancels_instead_of_reapplying(self):
        lines = wrap_ansi("\x1b[1mone two\x1b[22m three four", 8)
        # Bold ended on line one: nothing to reset or re-apply afterwards.
        assert lines == ["\x1b[1mone two\x1b[22m", "three", "four"]

    def test_style_still_carried_across_wrap(self):
        lines = wrap_ansi("\x1b[1mone two three\x1b[22m", 8)
        assert lines[0].endswith("\x1b[0m")
        assert lines[1].startswith("\x1b[1m")

    def test_truecolor_zero_component_is_not_a_reset(self):
        red = "\x1b[38;2;255;0;0m"
        assert _get_active_codes(split_ansi(f"\x1b[1m{red}x")) == f"\x1b[1m{red}"

    def test_hyperlink_is_closed_and_reopened_across_wrap(self):
        text = make_link("https://example.com", "click this long link") + " after"
        first, second, *_ = wrap_ansi(text, 11)
        assert OSC8_CLOSE in first
        assert "https://example.com" in second

    def test_break_words_keeps_whitespace_and_fills_lines(self):
        lines = wrap_ansi("    x = compute(alpha, beta)", 10, break_words=True)
        assert lines == ["    x = co", "mpute(alph", "a, beta)"]


class TestBlockWrapping:
    @pytest.mark.parametrize("width", [30, 45, 70])
    def test_every_line_fits(self, width):
        for line in render(DOC, width).splitlines():
            assert visible_length(line) <= width, repr(line)

    @pytest.mark.parametrize("width", [30, 45])
    def test_words_are_never_split(self, width):
        seen = tokens(render(DOC, width))
        for word in WORDS.split():
            assert word in seen or f"{word}." in seen, word

    def test_list_continuation_hangs_under_text(self):
        out = visible(render(f"- {WORDS}", 30))
        first, second, *_ = out.splitlines()
        assert first.startswith("• ")
        assert second.startswith("  ") and not second.startswith("   ")

    def test_h1_underline_spans_widest_wrapped_line(self):
        lines = visible(render(f"# {WORDS}", 30)).splitlines()
        assert set(lines[-1]) == {"═"} and len(lines[-1]) <= 30
        assert len(lines) > 2

    def test_think_lines_keep_their_gutter(self):
        lines = visible(render(f"<think>\n{WORDS}\n</think>", 30)).splitlines()
        body = [ln for ln in lines if not ln.startswith(("┌", "└"))]
        assert len(body) > 1 and all(ln.startswith("│ ") for ln in body)

    def test_long_code_lines_wrap_with_marker(self):
        code = "x = '" + "y" * 60 + "'"
        out = visible(
            render(f"```python\n{code}\n```", 30, features=RenderFeatures(clipboard=False))
        )
        code_lines = [ln for ln in out.splitlines() if "yyy" in ln or "x = " in ln]
        assert len(code_lines) > 1
        assert all(ln.startswith(CODE_CONTINUATION) for ln in code_lines[1:])
        joined = "".join(ln.removeprefix(CODE_CONTINUATION).rstrip() for ln in code_lines)
        assert joined == code

    def test_wrap_text_feature_disables_wrapping(self):
        out = render(f"{WORDS}\n\n- {WORDS}", 30, features=RenderFeatures(wrap_text=False))
        assert WORDS in visible(out)


class TestLiveWidth:
    def test_unpinned_width_follows_the_terminal(self, monkeypatch):
        sizes = iter([80, 30])
        monkeypatch.setattr(Renderer, "_detect_width", staticmethod(lambda: next(sizes)))
        renderer = Renderer(output=StringIO())
        assert renderer.width == 80
        assert renderer.width == 30

    def test_new_blocks_wrap_to_the_resized_width(self, monkeypatch):
        size = {"cols": 100}
        monkeypatch.setattr(Renderer, "_detect_width", staticmethod(lambda: size["cols"]))
        out = StringIO()
        renderer = Renderer(output=out)
        parser = Parser()
        renderer.render_all(parser.parse_line(WORDS))
        renderer.render_all(parser.parse_line(""))
        size["cols"] = 30
        renderer.render_all(parser.parse_line(WORDS))
        renderer.render_all(parser.finalize())
        first, *rest = [ln for ln in out.getvalue().splitlines() if ln]
        assert visible(first) == WORDS  # rendered before the resize
        assert rest and all(visible_length(ln) <= 30 for ln in rest)

    def test_code_block_keeps_its_width_through_a_resize(self, monkeypatch):
        size = {"cols": 60}
        monkeypatch.setattr(Renderer, "_detect_width", staticmethod(lambda: size["cols"]))
        out = StringIO()
        renderer = Renderer(output=out, features=RenderFeatures(clipboard=False))
        parser = Parser()
        renderer.render_all(parser.parse_line("```python"))
        size["cols"] = 40
        for line in ("x = 1", "```"):
            renderer.render_all(parser.parse_line(line))
        borders = [visible_length(ln) for ln in out.getvalue().splitlines() if "─" in ln]
        assert borders == [60, 60]

    def test_max_width_caps_live_width(self, monkeypatch):
        monkeypatch.setattr(Renderer, "_detect_width", staticmethod(lambda: 200))
        assert Renderer(output=StringIO(), max_width=100).width == 100

    def test_set_width_none_unpins(self, monkeypatch):
        monkeypatch.setattr(Renderer, "_detect_width", staticmethod(lambda: 77))
        renderer = Renderer(output=StringIO(), width=40)
        assert renderer.width == 40
        renderer.set_width(None)
        assert renderer.width == 77


class TestRenderMarkdownLines:
    @pytest.mark.parametrize("width", [30, 60])
    def test_lines_fit_width(self, width):
        lines = render_markdown_lines(DOC, width)
        assert lines and all(visible_length(ln) <= width for ln in lines)

    def test_never_emits_clipboard_escape(self):
        lines = render_markdown_lines("```python\nx = 1\n```", 40)
        assert not any(re.search(r"\x1b\]52;", ln) for ln in lines)
