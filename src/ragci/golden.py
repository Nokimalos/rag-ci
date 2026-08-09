"""Golden cases: questions anchored to document passages, stored as JSONL."""

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Passage(BaseModel):
    doc_id: str
    char_start: int | None = None
    char_end: int | None = None
    text: str


class GoldenCase(BaseModel):
    id: str
    question: str
    required_passages: list[Passage] = Field(min_length=1)
    reference_answer: str | None = None
    multi_hop: bool = False
    provenance: str = "manual"
    reviewed_at: str | None = None
    strata: dict[str, Any] = Field(default_factory=dict)


def load_golden(path: Path) -> Iterator[GoldenCase]:
    """Stream cases one line at a time; golden sets are expected to grow large."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield GoldenCase.model_validate_json(line)


def save_golden(path: Path, cases: Iterable[GoldenCase]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json(exclude_none=True) + "\n")


def golden_hash(path: Path) -> str:
    """Content hash used to detect that a baseline no longer describes this golden set.

    Order-insensitive: reordering lines is not a semantic change.
    """
    digests = sorted(
        hashlib.sha256(
            json.dumps(case.model_dump(exclude={"reviewed_at"}), sort_keys=True).encode()
        ).hexdigest()
        for case in load_golden(path)
    )
    return hashlib.sha256("".join(digests).encode()).hexdigest()
