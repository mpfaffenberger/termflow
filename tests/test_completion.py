"""Tests for the minimal completion protocol."""

import pytest

from termflow.tui.completion import (
    CompleteEvent,
    Completer,
    Completion,
    Document,
    merge_completers,
)


class TestDocument:
    def test_cursor_defaults_to_end(self):
        doc = Document("hello")
        assert doc.cursor_position == 5
        assert doc.text_before_cursor == "hello"
        assert doc.text_after_cursor == ""

    def test_explicit_cursor_slices(self):
        doc = Document("hello world", cursor_position=5)
        assert doc.text_before_cursor == "hello"
        assert doc.text_after_cursor == " world"

    def test_word_before_cursor(self):
        assert Document("/model gpt").get_word_before_cursor() == "gpt"
        assert Document("open @src/ma").get_word_before_cursor(WORD=True) == "@src/ma"
        assert Document("trailing space ").get_word_before_cursor() == ""
        assert Document("").get_word_before_cursor() == ""

    def test_word_before_cursor_narrow_stops_at_symbols(self):
        assert Document("a=b").get_word_before_cursor() == "b"
        # But path-ish characters stay part of the word.
        assert Document("path/to/file").get_word_before_cursor() == "path/to/file"


class TestCompletion:
    def test_positive_start_position_rejected(self):
        with pytest.raises(ValueError):
            Completion("x", start_position=1)

    def test_defaults(self):
        c = Completion("x")
        assert c.start_position == 0
        assert c.display is None


class _Static(Completer):
    def __init__(self, *words):
        self.words = words

    def get_completions(self, document, _complete_event):
        prefix = document.get_word_before_cursor()
        for word in self.words:
            if word.startswith(prefix):
                yield Completion(word, start_position=-len(prefix))


class TestMerge:
    def test_merge_preserves_order(self):
        merged = merge_completers([_Static("alpha"), _Static("also", "beta")])
        results = [c.text for c in merged.get_completions(Document("al"), CompleteEvent())]
        assert results == ["alpha", "also"]

    def test_duck_typed_foreign_completer(self):
        class Foreign:  # no inheritance -- duck typing must suffice
            def get_completions(self, _document, _complete_event):
                yield Completion("quack")

        merged = merge_completers([Foreign()])
        results = list(merged.get_completions(Document(""), CompleteEvent()))
        assert results[0].text == "quack"
