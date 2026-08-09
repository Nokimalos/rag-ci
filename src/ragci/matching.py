"""Does a retrieved chunk cover a required passage?

Ground truth is anchored to document passages rather than chunk identifiers, so that
changing the chunking strategy never invalidates a golden set.
"""

import re

from ragci.contract import Chunk
from ragci.golden import Passage

DEFAULT_COVERAGE_THRESHOLD = 0.5

_WORD = re.compile(r"\w+")


def _has_offsets(chunk: Chunk, passage: Passage) -> bool:
    return None not in (chunk.char_start, chunk.char_end, passage.char_start, passage.char_end)


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def is_fallback_match(chunk: Chunk, passage: Passage) -> bool:
    """True when coverage has to be decided on text overlap instead of offsets."""
    return chunk.doc_id == passage.doc_id and not _has_offsets(chunk, passage)


def covers(chunk: Chunk, passage: Passage, threshold: float = DEFAULT_COVERAGE_THRESHOLD) -> bool:
    if chunk.doc_id != passage.doc_id:
        return False

    if _has_offsets(chunk, passage):
        span = passage.char_end - passage.char_start
        if span <= 0:
            return False
        overlap = min(chunk.char_end, passage.char_end) - max(chunk.char_start, passage.char_start)
        return overlap / span >= threshold

    required = _tokens(passage.text)
    if not required:
        return False
    return len(required & _tokens(chunk.text)) / len(required) >= threshold
