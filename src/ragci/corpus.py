"""Reading a corpus without ever holding it in memory."""

import random
from collections.abc import Callable, Iterable, Iterator, Sequence
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


CorpusSource = Iterable[Document] | Callable[[], Iterable[Document]]


def _reader(documents: CorpusSource) -> Callable[[], Iterable[Document]]:
    """Something that can be walked twice.

    A callable is genuinely re-read, holding nothing but identifiers between passes.
    Anything else may be a one-shot iterator, so it is materialised once — walking an
    exhausted generator a second time would silently yield nothing and look like the
    corpus had changed underneath us.
    """
    if callable(documents):
        return documents
    materialised = list(documents)
    return lambda: materialised


def sample_with_report(
    documents: CorpusSource,
    *,
    n: int,
    keys: Sequence[str] = ("source",),
    seed: int = 0,
) -> tuple[list[Document], SampleReport]:
    """Sample n documents, guaranteeing every stratum appears at least once.

    Proportional allocation alone lets a source holding 0.1% of the corpus be sampled
    out of existence — which is precisely the case a golden set most needs covered.

    Pass a **callable** returning an iterable — `lambda: load_corpus(path)` — and the
    corpus is read twice while only identifiers are held between the passes: choosing
    what to sample needs a doc_id and a stratum key per document, never the text. On a
    million documents that is the difference between tens of megabytes and the whole
    corpus resident. Passing an iterable directly still works and still materialises
    everything; it is the convenient path, not the scalable one.
    """
    read = _reader(documents)

    buckets: dict[tuple[str, ...], list[str]] = {}
    total = 0
    for document in read():
        total += 1
        buckets.setdefault(stratum_key(document, keys), []).append(document.doc_id)

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

    strata = len(buckets)
    chosen: list[str] = []
    for key in order:
        chosen.extend(rng.sample(sorted(buckets[key]), allocation[key]))
    buckets.clear()  # the identifiers are no longer needed; only `chosen` is

    # Second pass: materialise only the documents that were selected. Collected by
    # doc_id, then re-ordered to match `chosen` so the result does not depend on the
    # order the corpus happens to be stored in.
    wanted = set(chosen)
    found: dict[str, Document] = {}
    for document in read():
        if document.doc_id in wanted and document.doc_id not in found:
            found[document.doc_id] = document
    sample = [found[doc_id] for doc_id in chosen if doc_id in found]

    if len(sample) != len(chosen):
        raise CorpusError(
            "the corpus changed between the two passes: "
            f"{len(chosen) - len(sample)} sampled document(s) were not found the second "
            "time. Sample from a snapshot rather than a corpus being written to."
        )

    report = SampleReport(
        total=total,
        strata=strata,
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
