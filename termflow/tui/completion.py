"""A minimal completion protocol for terminal line editors.

Deliberately shaped like the prompt_toolkit completion API (the subset
real completers actually use), so migrating a completer is an import
swap: ``Document`` carries the buffer text + cursor, completers yield
``Completion`` objects with a relative ``start_position``, and
``merge_completers`` chains stacks together.

The protocol is duck-typed on purpose: any object with ``text`` /
``cursor_position`` works as a document, and anything yielding objects
with ``text`` / ``start_position`` works as a completer -- so foreign
completer implementations keep working without inheriting from
:class:`Completer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass
class Document:
    """An immutable snapshot of a text buffer with a cursor."""

    text: str = ""
    cursor_position: int | None = None

    def __post_init__(self) -> None:
        if self.cursor_position is None:
            self.cursor_position = len(self.text)

    @property
    def text_before_cursor(self) -> str:
        """Everything up to (not including) the cursor."""
        return self.text[: self.cursor_position]

    @property
    def text_after_cursor(self) -> str:
        """Everything from the cursor onwards."""
        return self.text[self.cursor_position :]

    def get_word_before_cursor(self, WORD: bool = False) -> str:
        """The whitespace-delimited word immediately before the cursor.

        ``WORD`` matches the prompt_toolkit flag name: when True the word
        is delimited only by whitespace; when False it stops at
        non-alphanumeric characters too.
        """
        before = self.text_before_cursor
        if not before or before[-1].isspace():
            return ""
        word = before.split()[-1]
        if WORD:
            return word
        # Narrow mode: trim back to the last non-word character.
        for index in range(len(word) - 1, -1, -1):
            if not (word[index].isalnum() or word[index] in "_-./@"):
                return word[index + 1 :]
        return word


@dataclass
class CompleteEvent:
    """Why completions were requested (parity stub; rarely inspected)."""

    text_inserted: bool = False
    completion_requested: bool = True


@dataclass
class Completion:
    """One completion candidate.

    ``start_position`` is relative to the cursor and non-positive: it
    says how many characters before the cursor the replacement starts.
    """

    text: str
    start_position: int = 0
    display: str | None = None
    display_meta: str | None = None

    def __post_init__(self) -> None:
        if self.start_position > 0:
            raise ValueError("start_position must be <= 0")


class Completer:
    """Base class for completers. Subclass and implement get_completions."""

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass
class _MergedCompleter(Completer):
    completers: list = field(default_factory=list)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        for completer in self.completers:
            yield from completer.get_completions(document, complete_event)


def merge_completers(completers: Iterable) -> Completer:
    """Chain multiple completers into one (yielding in order)."""
    return _MergedCompleter(list(completers))
