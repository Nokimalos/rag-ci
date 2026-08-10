"""Execute golden cases against an adapter and summarise the result."""

import asyncio
import inspect
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from ragci.contract import RetrievalTrace
from ragci.golden import GoldenCase
from ragci.judge import Judge, Verdict, citation_accuracy
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
    verdict: Verdict | None = None


class Timings(BaseModel):
    """Kept apart from metrics: timings vary run to run and must not break determinism.

    None, not 0.0, is the default: save_json drops this block, so a reloaded record has
    no timings to report. Zeros would have been indistinguishable from a genuinely
    instant run, and the pull request comment said `latency p50 0 ms` because of it.
    """

    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None


class RunRecord(BaseModel):
    golden_hash: str
    config: dict[str, Any]
    primary_metric: str
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    case_results: list[CaseResult] = Field(default_factory=list)
    error_rate: float = 0.0
    valid: bool = True
    degraded_matching: bool = False
    judged: bool = False
    # Cost and token counts are deterministic, so unlike timings they stay in the diff.
    cost_usd_per_query: float | None = None
    tokens_per_query: float | None = None
    timings: Timings = Field(default_factory=Timings)


async def _call(func, *args):
    """Call an adapter method whether it is sync or async.

    Most modern RAG stacks are async — vector-db clients, FastAPI handlers, anything
    doing network I/O. Passing a coroutine function to to_thread returns the coroutine
    unawaited, which surfaces as an unrelated Pydantic validation error several frames
    later. Ask first.
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args)
    # Synchronous adapters run off the event loop so one slow call cannot block the rest.
    return await asyncio.to_thread(func, *args)


async def _run_one(adapter, case: GoldenCase, index, config, semaphore) -> CaseResult:
    async with semaphore:
        try:
            trace = await _call(adapter.retrieve, case.question, index, config)
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
    judge: Judge | None = None,
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

    judged = False
    if judge is not None and hasattr(adapter, "answer"):
        for case, result in zip(cases, results, strict=True):
            if result.status != "ok" or result.trace is None:
                continue
            try:
                answer = await _call(adapter.answer, case.question, result.trace, config)
                verdict = judge.assess(case.question, answer, result.trace.all_chunks)
            except Exception:  # noqa: BLE001 - a judge outage is not a retrieval failure
                verdict = None
            if verdict is None:
                continue
            judged = True
            result.verdict = verdict
            if verdict.faithfulness is not None:
                result.scores["faithfulness"] = verdict.faithfulness
                per_metric.setdefault("faithfulness", []).append(verdict.faithfulness)
            accuracy = citation_accuracy(answer, verdict, result.trace.all_chunks)
            if accuracy is not None:
                result.scores["citation_accuracy"] = accuracy
                per_metric.setdefault("citation_accuracy", []).append(accuracy)

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
        judged=judged,
        cost_usd_per_query=float(np.mean(costs)) if costs else None,
        tokens_per_query=float(np.mean(token_counts)) if token_counts else None,
        timings=Timings(
            latency_ms_p50=float(np.percentile(latencies, 50)),
            latency_ms_p95=float(np.percentile(latencies, 95)),
        ),
    )
