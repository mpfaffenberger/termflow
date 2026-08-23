"""Tests for termflow.diff — parsing, rendering, theming, streaming."""

import pytest

from termflow.diff import (
    DiffRenderer,
    DiffStream,
    DiffTheme,
    brighten_hex,
    classify_line,
    detect_language,
    parse_diff,
    render_diff,
)

SAMPLE_DIFF = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n context = True\n-old = 1\n+new = 2\n"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestClassifyLine:
    def test_add(self):
        line = classify_line("+new = 2")
        assert line.kind == "add"
        assert line.content == "new = 2"

    def test_remove(self):
        line = classify_line("-old = 1")
        assert line.kind == "remove"
        assert line.content == "old = 1"

    def test_context_strips_leading_space(self):
        line = classify_line(" context = True")
        assert line.kind == "context"
        assert line.content == "context = True"

    def test_context_without_space(self):
        line = classify_line("plain")
        assert line.kind == "context"
        assert line.content == "plain"

    @pytest.mark.parametrize(
        "raw",
        ["--- a/f.py", "+++ b/f.py", "@@ -1 +1 @@", "diff --git a/f b/f", "index abc..def"],
    )
    def test_headers(self, raw):
        assert classify_line(raw).kind == "header"

    def test_plus_plus_plus_is_header_not_add(self):
        # +++ must win over the bare + marker.
        assert classify_line("+++ b/f.py").kind == "header"


class TestParseDiff:
    def test_empty(self):
        assert parse_diff("") == []

    def test_trailing_newline_dropped(self):
        lines = parse_diff("-a\n+b\n")
        assert len(lines) == 2

    def test_kinds(self):
        kinds = [line.kind for line in parse_diff(SAMPLE_DIFF)]
        assert kinds == ["header", "header", "header", "context", "remove", "add"]


class TestDetectLanguage:
    def test_python(self):
        assert detect_language(SAMPLE_DIFF) == "py"

    def test_javascript(self):
        assert detect_language("--- a/app.js\n+++ b/app.js\n") == "js"

    def test_fallback(self):
        assert detect_language("-old\n+new") == "text"


# ---------------------------------------------------------------------------
# brighten_hex
# ---------------------------------------------------------------------------


class TestBrightenHex:
    def test_no_change(self):
        assert brighten_hex("#808080", 0.0) == "#808080"

    def test_clamp_max(self):
        assert brighten_hex("#ffffff", 1.0) == "#ffffff"

    def test_clamp_min(self):
        assert brighten_hex("#000000", -1.0) == "#000000"

    def test_invalid(self):
        with pytest.raises(ValueError):
            brighten_hex("#ff", 0.5)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class TestDiffRenderer:
    def test_headers_skipped_by_default(self):
        out = DiffRenderer().render(SAMPLE_DIFF)
        assert "@@" not in out
        assert "a/f.py" not in out

    def test_headers_dimmed_when_shown(self):
        out = DiffRenderer(show_headers=True).render(SAMPLE_DIFF)
        assert "\x1b[2m--- a/f.py" in out

    def test_backgrounds_applied(self):
        theme = DiffTheme(addition="#002200", deletion="#220000")
        out = DiffRenderer(theme=theme).render(SAMPLE_DIFF)
        assert "\x1b[48;2;0;34;0m" in out  # addition bg
        assert "\x1b[48;2;34;0;0m" in out  # deletion bg

    def test_context_has_no_background(self):
        out = DiffRenderer().render(" context only\n")
        assert "\x1b[48;2" not in out

    def test_markers_present(self):
        out = DiffRenderer().render(SAMPLE_DIFF)
        assert "+ " in out
        assert "- " in out

    def test_no_trailing_newline(self):
        assert not DiffRenderer().render(SAMPLE_DIFF).endswith("\n")

    def test_blank_line_renders_empty(self):
        out = DiffRenderer().render("-removed\n\n+added")
        assert "\n\n" in out

    def test_background_survives_resets(self):
        theme = DiffTheme(addition="#002200")
        out = DiffRenderer(theme=theme).render("+x = 1\n")
        bg = "\x1b[48;2;0;34;0m"
        # The bg is re-asserted after the last SGR before content ends.
        assert out.count(bg) >= 2

    def test_line_tints_shift_truecolor_foregrounds(self):
        theme = DiffTheme(addition="#002200", line_tints={"added": (100, 0, 0)})
        plain = DiffRenderer(theme=DiffTheme(addition="#002200")).render("+x = 1\n")
        tinted = DiffRenderer(theme=theme).render("+x = 1\n")
        assert plain != tinted

    def test_tints_adopted_from_highlighter(self):
        from termflow.syntax import Highlighter

        highlighter = Highlighter()
        highlighter.diff_line_tints = {"added": (10, 10, 10)}
        renderer = DiffRenderer(highlighter=highlighter)
        assert renderer.theme.line_tints == {"added": (10, 10, 10)}

    def test_render_diff_convenience(self):
        assert isinstance(render_diff(SAMPLE_DIFF), str)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestDiffStream:
    def test_stream_matches_block_render(self):
        renderer = DiffRenderer()
        stream = DiffStream(DiffRenderer())
        chunks = [SAMPLE_DIFF[i : i + 7] for i in range(0, len(SAMPLE_DIFF), 7)]
        streamed = "".join(stream.feed(chunk) for chunk in chunks) + stream.close()
        assert streamed.rstrip("\n") == renderer.render(SAMPLE_DIFF)

    def test_partial_lines_buffer(self):
        stream = DiffStream()
        assert stream.feed("+incomp") == ""
        assert stream.feed("lete\n") != ""

    def test_close_flushes_tail(self):
        stream = DiffStream()
        stream.feed("+tail with no newline")
        out = stream.close()
        assert "tail with no newline" in out
        assert not out.endswith("\n")

    def test_close_idempotent_when_empty(self):
        stream = DiffStream()
        assert stream.close() == ""

    def test_language_sniffed_from_headers(self):
        stream = DiffStream()
        stream.feed("--- a/f.py\n+++ b/f.py\n")
        assert stream._language == "py"

    def test_explicit_language_wins(self):
        stream = DiffStream(language="rust")
        stream.feed("--- a/f.py\n+++ b/f.py\n")
        assert stream._language == "rust"

    def test_multi_file_language_switch(self):
        stream = DiffStream()
        stream.feed("--- a/f.py\n+++ b/f.py\n+x = 1\n")
        stream.feed("--- a/g.rs\n+++ b/g.rs\n+let y = 2;\n")
        assert stream._language == "rs"
