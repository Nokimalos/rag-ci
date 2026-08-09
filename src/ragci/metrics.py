"""Tier-1 retrieval metrics: deterministic, free, and safe to gate on."""

import math
from collections.abc import Callable

from ragci.contract import RetrievalTrace
from ragci.golden import GoldenCase
from ragci.matching import DEFAULT_COVERAGE_THRESHOLD, covers

DEFAULT_K = 10


def _top_k(trace: RetrievalTrace, k: int):
    return trace.all_chunks[:k]


def recall_at_k(
    trace: RetrievalTrace, case: GoldenCase, k: int, threshold: float = DEFAULT_COVERAGE_THRESHOLD
) -> float:
    chunks = _top_k(trace, k)
    found = sum(
        any(covers(chunk, passage, threshold) for chunk in chunks)
        for passage in case.required_passages
    )
    return found / len(case.required_passages)


def all_passages_recall_at_k(
    trace: RetrievalTrace, case: GoldenCase, k: int, threshold: float = DEFAULT_COVERAGE_THRESHOLD
) -> float:
    """Partial credit hides multi-hop failures; this metric refuses to give any."""
    return 1.0 if recall_at_k(trace, case, k, threshold) == 1.0 else 0.0


def precision_at_k(
    trace: RetrievalTrace, case: GoldenCase, k: int, threshold: float = DEFAULT_COVERAGE_THRESHOLD
) -> float:
    chunks = _top_k(trace, k)
    if not chunks:
        return 0.0
    useful = sum(
        any(covers(chunk, passage, threshold) for passage in case.required_passages)
        for chunk in chunks
    )
    return useful / len(chunks)


def mrr(
    trace: RetrievalTrace, case: GoldenCase, k: int, threshold: float = DEFAULT_COVERAGE_THRESHOLD
) -> float:
    for rank, chunk in enumerate(_top_k(trace, k), start=1):
        if any(covers(chunk, passage, threshold) for passage in case.required_passages):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    trace: RetrievalTrace, case: GoldenCase, k: int, threshold: float = DEFAULT_COVERAGE_THRESHOLD
) -> float:
    gains = [
        1.0 if any(covers(chunk, passage, threshold) for passage in case.required_passages) else 0.0
        for chunk in _top_k(trace, k)
    ]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_hits = min(k, len(case.required_passages))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


TIER1_METRICS: dict[str, Callable[..., float]] = {
    "recall": recall_at_k,
    "all_passages_recall": all_passages_recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr,
    "ndcg": ndcg_at_k,
}


def parse_metric_name(name: str) -> tuple[str, int]:
    """`"recall@10"` -> `("recall", 10)`. A bare name uses the default k."""
    base, _, suffix = name.partition("@")
    if base not in TIER1_METRICS:
        raise ValueError(f"unknown metric: {base}")
    return base, int(suffix) if suffix else DEFAULT_K
