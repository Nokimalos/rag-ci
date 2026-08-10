"""Turning an existing question/answer set into passage-anchored golden cases.

Ground truth in rag-ci points at document passages by character offset, which is what
lets a golden set survive a change of chunk size. Teams that already have a Q/A set have
answers, not offsets, and until now `golden gen` was the only way in — starting from
scratch, with an API key.

Nothing here guesses. An answer that appears in one place becomes a passage; anything
ambiguous or absent is reported for a human to settle. A confidently wrong offset is
worse than no offset: it silently teaches the gate to reward the wrong retrieval.
"""

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ragci.corpus import Document
from ragci.golden import GoldenCase, Passage

MIN_FUZZY_OVERLAP = 0.5
MAX_SUGGESTIONS = 3


class QAPair(BaseModel):
    """What people already have: a question and the text of its answer."""

    id: str
    question: str
    answer: str


class AnchorOutcome(BaseModel):
    pair: QAPair
    status: Literal["anchored", "ambiguous", "not_found"]
    case: GoldenCase | None = None
    # Where the answer might be, for the two statuses a human has to settle.
    candidates: list[Passage] = []
    note: str = ""


def _normalise(text: str) -> tuple[str, list[int]]:
    """Case-folded, whitespace-collapsed text, plus each character's original index.

    The index is the whole point: matching has to tolerate the reflowed whitespace and
    casing that survive a copy-paste, while the offsets it reports must address the
    document exactly as stored.
    """
    out: list[str] = []
    index: list[int] = []
    previous_space = True  # leading whitespace is dropped

    for position, character in enumerate(text):
        if character.isspace():
            if previous_space:
                continue
            out.append(" ")
            index.append(position)
            previous_space = True
        else:
            out.append(character.lower())
            index.append(position)
            previous_space = False

    if out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def find_occurrences(document: Document, answer: str) -> list[Passage]:
    """Every place the answer appears in the document, as offsets into the stored text."""
    haystack, index = _normalise(document.text)
    needle, _ = _normalise(answer)
    if not needle:
        return []

    found: list[Passage] = []
    start = haystack.find(needle)
    while start != -1:
        char_start = index[start]
        char_end = index[start + len(needle) - 1] + 1
        found.append(
            Passage(
                doc_id=document.doc_id,
                char_start=char_start,
                char_end=char_end,
                text=document.text[char_start:char_end],
            )
        )
        start = haystack.find(needle, start + 1)
    return found


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2}


def _sentences(text: str) -> Iterator[tuple[int, int]]:
    """Sentence spans. A lone newline is *not* a boundary — documents arrive reflowed,
    and splitting on one cuts sentences in half, which loses the very passage a
    paraphrase would have matched. Only terminal punctuation and blank lines divide."""
    start = 0
    for match in re.finditer(r"[.!?]\s+|\n\s*\n", text):
        if match.end() > start:
            yield start, match.start() + 1
        start = match.end()
    if start < len(text):
        yield start, len(text)


def suggest_passages(document: Document, answer: str) -> list[Passage]:
    """Sentences that look like the answer, for when it is paraphrased rather than quoted.

    Returned as suggestions only. Deciding that a paraphrase is the same claim is a
    judgement, and the tool does not get to make it.
    """
    wanted = _tokens(answer)
    if not wanted:
        return []

    scored: list[tuple[float, Passage]] = []
    for start, end in _sentences(document.text):
        found = _tokens(document.text[start:end])
        if not found:
            continue
        shared = len(wanted & found)
        # F1 of both directions. Scoring only "how much of the answer is here" punishes
        # a verbose paraphrase, which is exactly the shape a paraphrase takes; scoring
        # only "how much of this sentence is in the answer" lets a three-word sentence
        # win by default. Neither alone is the question being asked.
        precision, recall = shared / len(found), shared / len(wanted)
        overlap = 2 * precision * recall / (precision + recall) if shared else 0.0
        if overlap >= MIN_FUZZY_OVERLAP:
            scored.append(
                (
                    overlap,
                    Passage(
                        doc_id=document.doc_id,
                        char_start=start,
                        char_end=end,
                        text=document.text[start:end],
                    ),
                )
            )
    scored.sort(key=lambda pair: (-pair[0], pair[1].char_start))
    return [passage for _, passage in scored[:MAX_SUGGESTIONS]]


def anchor(pairs: Sequence[QAPair], documents: Iterable[Document]) -> list[AnchorOutcome]:
    """Locate every answer in the corpus in a single pass over it.

    The corpus is streamed and the pairs are held in memory, not the other way round:
    a Q/A set is small and a corpus is not.
    """
    exact: dict[str, list[Passage]] = {pair.id: [] for pair in pairs}
    fuzzy: dict[str, list[Passage]] = {pair.id: [] for pair in pairs}

    for document in documents:
        for pair in pairs:
            hits = find_occurrences(document, pair.answer)
            if hits:
                exact[pair.id].extend(hits)
            elif not exact[pair.id]:  # only worth suggesting while nothing exact is known
                fuzzy[pair.id].extend(suggest_passages(document, pair.answer))

    outcomes: list[AnchorOutcome] = []
    for pair in pairs:
        hits = exact[pair.id]
        if len(hits) == 1:
            outcomes.append(
                AnchorOutcome(
                    pair=pair,
                    status="anchored",
                    case=GoldenCase(
                        id=pair.id,
                        question=pair.question,
                        required_passages=hits,
                        reference_answer=pair.answer,
                        provenance="anchored",
                    ),
                )
            )
        elif hits:
            places = ", ".join(sorted({p.doc_id for p in hits}))
            outcomes.append(
                AnchorOutcome(
                    pair=pair,
                    status="ambiguous",
                    candidates=hits[:MAX_SUGGESTIONS],
                    note=f"the answer appears {len(hits)} times ({places})",
                )
            )
        else:
            suggestions = sorted(fuzzy[pair.id], key=lambda p: p.char_start)[:MAX_SUGGESTIONS]
            outcomes.append(
                AnchorOutcome(
                    pair=pair,
                    status="not_found",
                    candidates=suggestions,
                    note=(
                        "no exact match; the answer is probably paraphrased or synthesised"
                        if suggestions
                        else "no exact match and nothing close — is this the right corpus?"
                    ),
                )
            )
    return outcomes


def load_pairs(path: Path) -> list[QAPair]:
    """Read a Q/A JSONL. `id` is optional; `question` and `answer` are not."""
    pairs: list[QAPair] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        missing = [field for field in ("question", "answer") if not record.get(field)]
        if missing:
            raise ValueError(f"{path}:{number} has no {' or '.join(missing)}")
        pairs.append(
            QAPair(
                id=str(record.get("id") or f"q{number:04d}"),
                question=record["question"],
                answer=record["answer"],
            )
        )
    return pairs
