"""Reusing an evaluation instead of paying for it twice.

A sweep re-evaluates configurations across rungs, and re-running one after changing a
single parameter re-measures everything that did not change. With an LLM judge in the
loop that is slow *and* billed.

The whole design turns on one risk: a cache that serves a stale result is worse than no
cache at all, because this tool's output is a verdict people act on. So the key contains
everything that can change the answer, and when something cannot be captured in a key —
an embedding service that silently changed, an index rebuilt outside the adapter — the
answer is that caching is opt-in and the user decides when it is safe.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ragci.runner import RunRecord


class CacheStats(BaseModel):
    hits: int = 0
    misses: int = 0

    @property
    def saved(self) -> int:
        return self.hits


def fingerprint_file(path: Path) -> str:
    """A hash of the adapter's source. Edit the adapter and every entry stops matching."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def digest_cases(cases: Sequence[Any]) -> str:
    """Hash the cases in order — order decides which ones a rung evaluates."""
    hasher = hashlib.sha256()
    for case in cases:
        hasher.update(case.model_dump_json(exclude_none=True).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


class RunCache:
    """Content-addressed store of run records, keyed on everything that decides one."""

    def __init__(self, root: Path, *, fingerprint: str, version: str) -> None:
        self.root = Path(root)
        self._fingerprint = fingerprint
        self._version = version
        self.stats = CacheStats()

    def key(self, *, config: dict, cases: Sequence[Any], metrics: Sequence[str]) -> str:
        payload = json.dumps(
            {
                # The rag-ci version is part of the key: a metric whose definition changed
                # between releases would otherwise be served from an older one's results.
                "version": self._version,
                "adapter": self._fingerprint,
                "config": config,
                "cases": digest_cases(cases),
                "metrics": sorted(metrics),
            },
            sort_keys=True,
            default=repr,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> RunRecord | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            self.stats.misses += 1
            return None
        try:
            record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            # A truncated or older-format entry is a miss, not a crash. Re-measuring is
            # always available; a cache that can break a run is not worth having.
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return record

    def put(self, key: str, record: RunRecord) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump(exclude={"timings"})
        target = self.root / f"{key}.json"
        # Write then move: a run interrupted mid-write must not leave a half-entry that
        # a later run reads as a result.
        temporary = target.with_suffix(".json.partial")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
