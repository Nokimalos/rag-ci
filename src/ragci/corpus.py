"""Reading a corpus without ever holding it in memory."""

import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

TEXT_SUFFIXES = {".txt", ".md"}


class CorpusError(Exception):
    """The corpus could not be read. The message says what to do about it."""


class Document(BaseModel):
    doc_id: str = Field(min_length=1)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def _load_directory(root: Path) -> Iterator[Document]:
    found = False
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        found = True
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        relative = path.relative_to(root)
        yield Document(
            # POSIX-style so a corpus hashes identically on every platform.
            doc_id=relative.as_posix(),
            text=text,
            metadata={"source": relative.parent.as_posix()},
        )
    if not found:
        raise CorpusError(f"{root} contains no .txt or .md files")


def _load_jsonl(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            document = Document.model_validate_json(line)
            if document.text.strip():
                yield document


def load_corpus(path: Path) -> Iterator[Document]:
    """Stream a corpus from a directory of text files or a JSONL export."""
    path = Path(path)
    if not path.exists():
        raise CorpusError(f"{path} does not exist")
    if path.is_dir():
        yield from _load_directory(path)
    else:
        yield from _load_jsonl(path)


MISSING = "<missing>"


class SampleReport(BaseModel):
    total: int
    strata: int
    sampled: int
    per_stratum: dict[str, int]


def stratum_key(document: Document, keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(document.metadata.get(key, MISSING)) for key in keys)


def sample_with_report(
    documents: Iterable[Document],
    *,
    n: int,
    keys: Sequence[str] = ("source",),
    seed: int = 0,
) -> tuple[list[Document], SampleReport]:
    """Sample n documents, guaranteeing every stratum appears at least once.

    Proportional allocation alone lets a source holding 0.1% of the corpus be sampled
    out of existence — which is precisely the case a golden set most needs covered.
    """
    buckets: dict[tuple[str, ...], list[Document]] = {}
    total = 0
    for document in documents:
        total += 1
        buckets.setdefault(stratum_key(document, keys), []).append(document)

    if total == 0:
        raise CorpusError("cannot sample: the corpus contains no documents")

    rng = random.Random(seed)
    order = sorted(buckets)  # stable iteration regardless of insertion order

    # More strata than slots: n is a hard cap, so choose which strata are represented
    # rather than returning one document per stratum and blowing past n.
    if len(order) > n:
        order = sorted(rng.sample(order, n))

    # One slot each, then the remainder proportionally, then redistribute whatever
    # a small stratum could not absorb.
    allocation = {key: min(1, len(buckets[key])) for key in order}
    remaining = n - sum(allocation.values())
    if remaining > 0:
        weights = {key: len(buckets[key]) / total for key in order}
        for key in sorted(order, key=lambda k: weights[k], reverse=True):
            if remaining <= 0:
                break
            headroom = len(buckets[key]) - allocation[key]
            take = min(headroom, max(1, int(remaining * weights[key])), remaining)
            allocation[key] += take
            remaining -= take
        for key in order:  # spare capacity from strata that ran out
            if remaining <= 0:
                break
            take = min(len(buckets[key]) - allocation[key], remaining)
            allocation[key] += take
            remaining -= take

    sample: list[Document] = []
    for key in order:
        pool = sorted(buckets[key], key=lambda d: d.doc_id)
        sample.extend(rng.sample(pool, allocation[key]))

    report = SampleReport(
        total=total,
        strata=len(buckets),
        sampled=len(sample),
        per_stratum={"/".join(key): allocation[key] for key in order},
    )
    return sample, report


def stratified_sample(
    documents: Iterable[Document],
    *,
    n: int,
    keys: Sequence[str] = ("source",),
    seed: int = 0,
) -> list[Document]:
    return sample_with_report(documents, n=n, keys=keys, seed=seed)[0]
