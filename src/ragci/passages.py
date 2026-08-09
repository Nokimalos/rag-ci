"""Splitting a document into passages a question could be answered from."""

import re

from ragci.corpus import Document
from ragci.golden import Passage

MIN_PASSAGE_CHARS = 120
MAX_PASSAGE_CHARS = 1500

_BLANK_LINE = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _spans(text: str) -> list[tuple[int, int]]:
    """Paragraph spans as (start, end) offsets into `text`."""
    spans, cursor = [], 0
    for match in _BLANK_LINE.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span past surrounding whitespace, keeping offsets exact."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    starts = [start] + [m.end() for m in _SENTENCE_END.finditer(text, start, end)]
    spans = []
    for index, span_start in enumerate(starts):
        span_end = starts[index + 1] if index + 1 < len(starts) else end
        if span_start < span_end:
            spans.append((span_start, span_end))
    return spans


def _split_long(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Break an overlong span on sentence boundaries rather than mid-word.

    Sentences accumulate until the next one would exceed the limit — cutting only
    once the limit is already passed would make every piece overshoot it.
    """
    if end - start <= max_chars:
        return [(start, end)]

    pieces: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for span_start, span_end in _sentence_spans(text, start, end):
        if current is None:
            current = (span_start, span_end)
        elif span_end - current[0] <= max_chars:
            current = (current[0], span_end)
        else:
            pieces.append(current)
            current = (span_start, span_end)
    if current is not None:
        pieces.append(current)

    # A single sentence longer than the limit still has to be cut somewhere.
    bounded: list[tuple[int, int]] = []
    for piece_start, piece_end in pieces:
        while piece_end - piece_start > max_chars:
            bounded.append((piece_start, piece_start + max_chars))
            piece_start += max_chars
        if piece_end > piece_start:
            bounded.append((piece_start, piece_end))
    return bounded


def candidate_passages(
    document: Document,
    *,
    min_chars: int = MIN_PASSAGE_CHARS,
    max_chars: int = MAX_PASSAGE_CHARS,
) -> list[Passage]:
    """Passages long enough to answer a question, with offsets into the document."""
    text = document.text
    passages: list[Passage] = []

    for raw_start, raw_end in _spans(text):
        start, end = _trim(text, raw_start, raw_end)
        for piece_start, piece_end in _split_long(text, start, end, max_chars):
            piece_start, piece_end = _trim(text, piece_start, piece_end)
            if piece_end - piece_start < min_chars:
                continue
            passages.append(
                Passage(
                    doc_id=document.doc_id,
                    char_start=piece_start,
                    char_end=piece_end,
                    text=text[piece_start:piece_end],
                )
            )
    return passages
