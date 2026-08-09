"""Execute golden cases against an adapter and summarise the result."""

import asyncio
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from ragci.contract import RetrievalTrace
from ragci.golden import GoldenCase
from ragci.matching import is_fallback_match
from ragci.metrics import TIER1_METRICS, parse_metric_name
from ragci.stats import MetricSummary, bootstrap_ci

MAX_ERROR_RATE = 0.05


class CaseResult(BaseModel):
    case_id: str
    status: Literal["ok", "error"]
    error: str | None = None
    trace: RetrievalTrace | None = None
    scores: dict[str, float] = Field(default_factory=dict)


class Timings(BaseModel):
    """Kept apart from metrics: timings vary run to run and must not break determinism."""

    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0


class RunRecord(BaseModel):
    golden_hash: str
    config: dict[str, Any]
    primary_metric: str
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    case_results: list[CaseResult] = Field(default_factory=list)
    error_rate: float = 0.0
    valid: bool = True
    degraded_matching: bool = False
    # Cost and token counts are deterministic, so unlike timings they stay in the diff.
    cost_usd_per_query: float | None = None
    tokens_per_query: float | None = None
    timings: Timings = Field(default_factory=Timings)


async def _run_one(adapter, case: GoldenCase, index, config, semaphore) -> CaseResult:
    async with semaphore:
        try:
            # User adapters are ordinary synchronous code; keep the event loop free.
            trace = await asyncio.to_thread(adapter.retrieve, case.question, index, config)
        except Exception as exc:  # noqa: BLE001 - any adapter failure is a case error
            return CaseResult(case_id=case.id, status="error", error=f"{type(exc).__name__}: {exc}")
        return CaseResult(case_id=case.id, status="ok", trace=trace)


async def run_cases(
    adapter,
    cases: Sequence[GoldenCase],
    *,
    config: dict[str, Any],
    metric_names: Sequence[str],
    golden_hash: str,
    index: Any = None,
    concurrency: int = 8,
    seed: int = 0,
) -> RunRecord:
    cases = list(cases)
    if not cases:
        raise ValueError("cannot run with no cases in the golden set")

    semaphore = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(_run_one(adapter, case, index, config, semaphore) for case in cases)
    )

    degraded = False
    per_metric: dict[str, list[float]] = {name: [] for name in metric_names}
    for case, result in zip(cases, results, strict=True):
        if result.status != "ok" or result.trace is None:
            continue
        for chunk in result.trace.all_chunks:
            degraded = degraded or any(
                is_fallback_match(chunk, passage) for passage in case.required_passages
            )
        for name in metric_names:
            base, k = parse_metric_name(name)
            score = TIER1_METRICS[base](result.trace, case, k)
            result.scores[name] = score
            per_metric[name].append(score)

    errors = sum(result.status == "error" for result in results)
    error_rate = errors / len(cases)

    traces = [result.trace for result in results if result.trace is not None]
    latencies = [trace.latency_ms for trace in traces] or [0.0]

    costs = [trace.cost_usd for trace in traces if trace.cost_usd is not None]
    token_counts = [
        trace.tokens.input_tokens + trace.tokens.output_tokens
        for trace in traces
        if trace.tokens is not None
    ]

    return RunRecord(
        golden_hash=golden_hash,
        config=dict(config),
        primary_metric=metric_names[0],
        metrics={
            name: bootstrap_ci(scores, seed=seed) for name, scores in per_metric.items() if scores
        },
        case_results=list(results),
        error_rate=error_rate,
        # Above the threshold the run says nothing about quality — it says the plumbing broke.
        valid=error_rate <= MAX_ERROR_RATE,
        degraded_matching=degraded,
        cost_usd_per_query=float(np.mean(costs)) if costs else None,
        tokens_per_query=float(np.mean(token_counts)) if token_counts else None,
        timings=Timings(
            latency_ms_p50=float(np.percentile(latencies, 50)),
            latency_ms_p95=float(np.percentile(latencies, 95)),
        ),
    )
