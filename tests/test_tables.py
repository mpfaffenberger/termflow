"""Regressions for table column alignment and terminal-cell geometry."""

from io import StringIO

import pytest
from wcwidth import wcswidth

from termflow import Parser, Renderer
from termflow.ansi import visible
from termflow.parser.events import TableSeparatorEvent
from termflow.render import RenderStyle
from termflow.render.table import render_table_complete


@pytest.mark.parametrize("outer_pipes", [False, True])
@pytest.mark.parametrize(
    ("separator", "expected"),
    [
        ("--- | :---: | ---:", ("none", "center", "right")),
        (":--- | --- | ---:", ("left", "none", "right")),
        (":---: | ---: | ---", ("center", "right", "none")),
        ("--- | --- | ---", ("none", "none", "none")),
    ],
)
def test_separator_preserves_every_column(separator, expected, outer_pipes):
    if outer_pipes:
        separator = f"| {separator} |"
    events = Parser().parse_document(f"| First | Second | Third |\n{separator}\n")
    event = next(event for event in events if isinstance(event, TableSeparatorEvent))
    assert event.alignments == expected


def test_mixed_alignment_through_parser_and_renderer():
    markdown = "| Left | Center | Right |\n| --- | :---: | ---: |\n| x | y | z |\n"
    output = StringIO()
    Renderer(output=output, width=80).render_all(Parser().parse_document(markdown))
    row = next(line for line in visible(output.getvalue()).splitlines() if "x" in line)
    assert row.split("│")[1:-1] == [" x    ", "   y    ", "     z "]


@pytest.mark.parametrize("cell", ["\u2705", "\u754c", "e\u0301", "\u2764\ufe0f"])
def test_right_aligned_unicode_cell_preserved_with_straight_borders(cell):
    markdown = f"| Left | Center | Right |\n| --- | :---: | ---: |\n| x | y | {cell} |\n"
    output = StringIO()
    Renderer(output=output, width=80).render_all(Parser().parse_document(markdown))
    lines = [line for line in visible(output.getvalue()).splitlines() if line.strip()]
    row = next(line for line in lines if cell in line)
    assert row.split("│")[-2] == " " * (6 - wcswidth(cell)) + cell + " "
    assert len({wcswidth(line) for line in lines}) == 1


def test_wrapped_wide_cells_keep_alignment_and_borders(monkeypatch):
    monkeypatch.delenv("TERMFLOW_MAX_TABLE_WIDTH", raising=False)
    cell = "\u2705" * 8
    lines = [
        visible(line)
        for line in render_table_complete(
            ["Left", "Center", "Right"],
            [["x", "y", cell]],
            ["none", "center", "right"],
            width=34,
            margin="",
            style=RenderStyle(),
        )
    ]
    body = lines[3:-1]
    assert len(body) > 1
    assert "".join(line.split("│")[-2].strip() for line in body) == cell
    assert {wcswidth(line) for line in lines} == {34}
