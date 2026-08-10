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
    # How many attempts this case needed. Above 1 means something is flaky, which the
    # report says out loud rather than hiding behind a result that eventually worked.
    attempts: int = 1


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
    # False means the run stopped early by policy (for example, a cost budget).
    # This is distinct from `valid`: the pipeline may be healthy while the sample is incomplete.
    complete: bool = True
    degraded_matching: bool = False
    judged: bool = False
    # Cost and token counts are deterministic, so unlike timings they stay in the diff.
    cost_usd_per_query: float | None = None
    tokens_per_query: float | None = None
    timings: Timings = Field(default_factory=Timings)

    @property
    def retried(self) -> int:
        """Cases that needed more than one attempt.

        Surfaced rather than swallowed: retries turn a flaky pipeline into a passing run,
        and a passing run that took three attempts per case is telling you something the
        metric cannot.
        """
        return sum(1 for result in self.case_results if result.attempts > 1)


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


RETRY_BACKOFF_SECONDS = 0.5


async def _run_one(
    adapter, case: GoldenCase, index, config, semaphore, retries: int = 0
) -> CaseResult:
    """Retrieve one case, retrying transient failures.

    A long sweep against a network-backed retriever will hit the occasional timeout, and
    without retries a handful of those pushes the run past its 5% error threshold and
    invalidates work that had nothing wrong with it.

    Every exception is retried, because an adapter can raise anything and guessing which
    failures are transient would be guessing. A deterministic failure therefore costs
    `retries` extra attempts before it is reported — bounded, and the price of not
    discarding a good run over a blip.
    """
    async with semaphore:
        last: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                trace = await _call(adapter.retrieve, case.question, index, config)
            except Exception as exc:  # noqa: BLE001 - any adapter failure is a case error
                last = exc
                if attempt <= retries:
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            return CaseResult(case_id=case.id, status="ok", trace=trace, attempts=attempt)

        return CaseResult(
            case_id=case.id,
            status="error",
            error=f"{type(last).__name__}: {last}",
            attempts=retries + 1,
        )


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
    retries: int = 0,
    max_cost: float | None = None,
) -> RunRecord:
    cases = list(cases)
    if not cases:
        raise ValueError("cannot run with no cases in the golden set")
    if max_cost is not None and max_cost <= 0:
        raise ValueError("max_cost must be greater than zero")

    semaphore = asyncio.Semaphore(concurrency)
    complete = True
    total_known_cost = 0.0
    case_costs: dict[str, float] = {}

    if max_cost is None:
        results = await asyncio.gather(
            *(_run_one(adapter, case, index, config, semaphore, retries) for case in cases)
        )
    else:
        # Cost is only known after an adapter call returns. Launching the full set
        # concurrently would allow every in-flight call to overshoot the budget, so a
        # budgeted run intentionally serializes paid work. The one unavoidable overshoot
        # is the call that first reports a total at or above the limit.
        results = []
        for position, case in enumerate(cases):
            result = await _run_one(adapter, case, index, config, semaphore, retries)
            results.append(result)
            if result.trace is not None and result.trace.cost_usd is not None:
                cost = result.trace.cost_usd
                case_costs[result.case_id] = case_costs.get(result.case_id, 0.0) + cost
                total_known_cost += cost
            if total_known_cost >= max_cost and position + 1 < len(cases):
                complete = False
                break

    if max_cost is None:
        for result in results:
            if result.trace is not None and result.trace.cost_usd is not None:
                case_costs[result.case_id] = result.trace.cost_usd
                total_known_cost += result.trace.cost_usd

    evaluated_cases = cases[: len(results)]
    degraded = False
    per_metric: dict[str, list[float]] = {name: [] for name in metric_names}
    for case, result in zip(evaluated_cases, results, strict=True):
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
        eligible = [
            (case, result)
            for case, result in zip(evaluated_cases, results, strict=True)
            if result.status == "ok" and result.trace is not None
        ]
        for position, (case, result) in enumerate(eligible):
            if max_cost is not None and total_known_cost >= max_cost:
                complete = False
                break
            try:
                answer = await _call(adapter.answer, case.question, result.trace, config)
                if answer.cost_usd is not None:
                    case_costs[result.case_id] = (
                        case_costs.get(result.case_id, 0.0) + answer.cost_usd
                    )
                    total_known_cost += answer.cost_usd
                verdict = judge.assess(case.question, answer, result.trace.all_chunks)
            except Exception:  # noqa: BLE001 - a judge outage is not a retrieval failure
                verdict = None
            if verdict is not None:
                judged = True
                result.verdict = verdict
                if verdict.faithfulness is not None:
                    result.scores["faithfulness"] = verdict.faithfulness
                    per_metric.setdefault("faithfulness", []).append(verdict.faithfulness)
                accuracy = citation_accuracy(answer, verdict, result.trace.all_chunks)
                if accuracy is not None:
                    result.scores["citation_accuracy"] = accuracy
                    per_metric.setdefault("citation_accuracy", []).append(accuracy)
            if (
                max_cost is not None
                and total_known_cost >= max_cost
                and position + 1 < len(eligible)
            ):
                complete = False
                break

    errors = sum(result.status == "error" for result in results)
    error_rate = errors / len(results) if results else 0.0

    traces = [result.trace for result in results if result.trace is not None]
    latencies = [trace.latency_ms for trace in traces] or [0.0]

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
        complete=complete,
        degraded_matching=degraded,
        judged=judged,
        cost_usd_per_query=(float(np.mean(list(case_costs.values()))) if case_costs else None),
        tokens_per_query=float(np.mean(token_counts)) if token_counts else None,
        timings=Timings(
            latency_ms_p50=float(np.percentile(latencies, 50)),
            latency_ms_p95=float(np.percentile(latencies, 95)),
        ),
    )
