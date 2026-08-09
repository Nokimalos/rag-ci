import pytest

from ragci.contract import Chunk, RetrievalTrace, Step
from ragci.golden import GoldenCase, Passage
from ragci.runner import run_cases
from tests.fixtures.reference_adapter import ReferenceRag


def _case(case_id: str, doc_id: str, start: int, end: int, text: str) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        question=text,
        required_passages=[Passage(doc_id=doc_id, char_start=start, char_end=end, text=text)],
    )


CASES = [
    _case("q1", "physics", 0, 45, "Gravity is the attraction between masses"),
    _case("q2", "biology", 0, 50, "Photosynthesis converts light into chemical energy"),
]


async def test_run_scores_every_case():
    record = await run_cases(
        ReferenceRag(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
    )
    assert len(record.case_results) == 2
    assert all(result.status == "ok" for result in record.case_results)


async def test_run_summarises_metrics_with_an_interval():
    record = await run_cases(
        ReferenceRag(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
    )
    summary = record.metrics["recall@5"]
    assert summary.n == 2
    assert summary.ci_low <= summary.mean <= summary.ci_high


async def test_a_raising_adapter_marks_the_case_as_error_not_zero():
    class Exploding:
        def retrieve(self, query, index, config):
            raise RuntimeError("provider exploded")

    record = await run_cases(
        Exploding(), CASES, config={}, metric_names=["recall@5"], golden_hash="h"
    )
    assert all(result.status == "error" for result in record.case_results)
    assert "provider exploded" in record.case_results[0].error
    assert record.error_rate == 1.0
    assert record.valid is False


async def test_errored_cases_are_excluded_from_metrics_rather_than_scored_zero():
    class HalfBroken:
        def __init__(self):
            self.calls = 0

        def retrieve(self, query, index, config):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return ReferenceRag().retrieve(query, index, config)

    record = await run_cases(
        HalfBroken(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        concurrency=1,
    )
    # One case scored, not two: a failure must never dilute the mean.
    assert record.metrics["recall@5"].n == 1
    assert record.error_rate == 0.5
    assert record.valid is False


async def test_a_run_below_the_error_threshold_stays_valid():
    record = await run_cases(
        ReferenceRag(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
    )
    assert record.error_rate == 0.0
    assert record.valid is True


async def test_degraded_matching_is_flagged_when_the_adapter_omits_offsets():
    class NoOffsets:
        def retrieve(self, query, index, config):
            return RetrievalTrace(
                steps=[Step(query=query, chunks=[Chunk(text=query, doc_id="physics")])]
            )

    record = await run_cases(
        NoOffsets(), CASES[:1], config={}, metric_names=["recall@5"], golden_hash="h"
    )
    assert record.degraded_matching is True


async def test_timings_are_reported_separately_from_metrics():
    record = await run_cases(
        ReferenceRag(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
    )
    assert record.timings.latency_ms_p95 >= record.timings.latency_ms_p50
    assert "latency" not in record.metrics


async def test_the_record_carries_the_golden_hash_and_config():
    record = await run_cases(
        ReferenceRag(),
        CASES,
        config={"top_k": 3},
        metric_names=["recall@5"],
        golden_hash="abc123",
    )
    assert record.golden_hash == "abc123"
    assert record.config == {"top_k": 3}


async def test_results_keep_golden_set_order_despite_concurrency():
    record = await run_cases(
        ReferenceRag(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        concurrency=8,
    )
    assert [result.case_id for result in record.case_results] == ["q1", "q2"]


async def test_cost_and_tokens_are_averaged_per_query_when_the_adapter_reports_them():
    from ragci.contract import TokenUsage

    class Metered:
        def retrieve(self, query, index, config):
            trace = ReferenceRag().retrieve(query, index, config)
            trace.cost_usd = 0.002
            trace.tokens = TokenUsage(input_tokens=300, output_tokens=100)
            return trace

    record = await run_cases(
        Metered(), CASES, config={"top_k": 5}, metric_names=["recall@5"], golden_hash="h"
    )
    assert record.cost_usd_per_query == pytest.approx(0.002)
    assert record.tokens_per_query == pytest.approx(400.0)


async def test_cost_and_tokens_are_none_when_the_adapter_reports_nothing():
    record = await run_cases(
        ReferenceRag(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
    )
    assert record.cost_usd_per_query is None
    assert record.tokens_per_query is None


async def test_an_empty_golden_set_is_rejected():
    with pytest.raises(ValueError, match="no cases"):
        await run_cases(ReferenceRag(), [], config={}, metric_names=["recall@5"], golden_hash="h")
