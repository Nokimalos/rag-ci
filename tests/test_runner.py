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


async def test_max_cost_stops_before_launching_another_paid_case():
    class Metered:
        def __init__(self):
            self.calls = 0

        def retrieve(self, query, index, config):
            self.calls += 1
            trace = ReferenceRag().retrieve(query, index, config)
            trace.cost_usd = 0.6
            return trace

    cases = CASES + [_case("q3", "physics", 0, 45, "Gravity is the attraction between masses")]
    adapter = Metered()
    record = await run_cases(
        adapter,
        cases,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        max_cost=1.0,
        concurrency=8,
    )
    assert adapter.calls == 2
    assert len(record.case_results) == 2
    assert record.complete is False
    assert record.valid is True
    assert record.cost_usd_per_query == pytest.approx(0.6)


async def test_max_cost_is_a_noop_when_the_adapter_reports_no_cost():
    record = await run_cases(
        ReferenceRag(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        max_cost=0.001,
    )
    assert len(record.case_results) == len(CASES)
    assert record.complete is True


async def test_answer_cost_is_included_in_unbudgeted_per_query_cost():
    from ragci.contract import Answer
    from tests.fakes import AnsweringRag, FakeJudge

    class Metered(AnsweringRag):
        def retrieve(self, query, index, config):
            trace = super().retrieve(query, index, config)
            trace.cost_usd = 0.2
            return trace

        def answer(self, query, trace, config):
            base = super().answer(query, trace, config)
            return Answer(
                text=base.text,
                cited_chunks=base.cited_chunks,
                latency_ms=base.latency_ms,
                cost_usd=0.3,
            )

    record = await run_cases(
        Metered(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        judge=FakeJudge(),
    )
    assert record.cost_usd_per_query == pytest.approx(0.5)


async def test_max_cost_counts_answer_cost_and_stops_before_the_next_answer():
    from ragci.contract import Answer
    from tests.fakes import AnsweringRag, FakeJudge

    class MeteredAnswers(AnsweringRag):
        def __init__(self):
            super().__init__()
            self.answer_calls = 0

        def answer(self, query, trace, config):
            self.answer_calls += 1
            base = super().answer(query, trace, config)
            return Answer(
                text=base.text,
                cited_chunks=base.cited_chunks,
                latency_ms=base.latency_ms,
                cost_usd=0.6,
            )

    cases = CASES + [_case("q3", "physics", 0, 45, "Gravity is the attraction between masses")]
    adapter = MeteredAnswers()
    record = await run_cases(
        adapter,
        cases,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        judge=FakeJudge(),
        max_cost=1.0,
    )
    assert adapter.answer_calls == 2
    assert record.complete is False
    assert record.cost_usd_per_query == pytest.approx(0.6)


async def test_an_empty_golden_set_is_rejected():
    with pytest.raises(ValueError, match="no cases"):
        await run_cases(ReferenceRag(), [], config={}, metric_names=["recall@5"], golden_hash="h")


async def test_a_judge_adds_tier_two_metrics():
    from tests.fakes import AnsweringRag, FakeJudge

    record = await run_cases(
        AnsweringRag(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        judge=FakeJudge(),
    )
    assert "faithfulness" in record.metrics
    assert record.judged is True


async def test_no_judge_means_no_tier_two_metrics():
    from tests.fakes import AnsweringRag

    record = await run_cases(
        AnsweringRag(), CASES, config={"top_k": 5}, metric_names=["recall@5"], golden_hash="h"
    )
    assert "faithfulness" not in record.metrics
    assert record.judged is False


async def test_an_adapter_without_answer_is_not_judged():
    from tests.fakes import FakeJudge

    # ReferenceRag retrieves but does not generate; asking for tier 2 must not crash.
    record = await run_cases(
        ReferenceRag(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        judge=FakeJudge(),
    )
    assert record.judged is False


async def test_a_judge_failure_does_not_invalidate_the_run():
    from tests.fakes import AnsweringRag, FakeJudge

    # Tier 1 measured fine; a judge outage must not turn that into an invalid run.
    record = await run_cases(
        AnsweringRag(),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        judge=FakeJudge(fail=True),
    )
    assert record.valid is True
    assert record.metrics["recall@5"].n == len(CASES)
    assert "faithfulness" not in record.metrics


async def test_citation_accuracy_is_absent_when_nothing_is_cited():
    from tests.fakes import AnsweringRag, FakeJudge

    record = await run_cases(
        AnsweringRag(cite=False),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        judge=FakeJudge(),
    )
    assert "faithfulness" in record.metrics
    assert "citation_accuracy" not in record.metrics


async def test_the_verdict_is_kept_on_the_case_result():
    from tests.fakes import AnsweringRag, FakeJudge

    record = await run_cases(
        AnsweringRag(),
        CASES[:1],
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        judge=FakeJudge(),
    )
    assert record.case_results[0].verdict is not None


class AsyncRag:
    """Most modern RAG stacks are async: vector-db clients, FastAPI, network calls."""

    def __init__(self, answering: bool = False):
        self._inner = ReferenceRag()
        if answering:
            self.answer = self._answer

    async def retrieve(self, query, index, config):
        return self._inner.retrieve(query, index, config)

    async def _answer(self, query, trace, config):
        from ragci.contract import Answer

        return Answer(text="Gravity.", cited_chunks=trace.all_chunks[:1], latency_ms=1.0)


async def test_an_async_retrieve_is_awaited():
    record = await run_cases(
        AsyncRag(), CASES, config={"top_k": 5}, metric_names=["recall@5"], golden_hash="h"
    )
    assert all(r.status == "ok" for r in record.case_results)
    assert record.metrics["recall@5"].n == len(CASES)


async def test_an_async_adapter_scores_the_same_as_a_sync_one():
    kwargs = dict(config={"top_k": 5}, metric_names=["recall@5"], golden_hash="h")
    sync = await run_cases(ReferenceRag(), CASES, **kwargs)
    asynchronous = await run_cases(AsyncRag(), CASES, **kwargs)
    assert sync.metrics["recall@5"].mean == asynchronous.metrics["recall@5"].mean


async def test_an_async_answer_is_awaited():
    from tests.fakes import FakeJudge

    record = await run_cases(
        AsyncRag(answering=True),
        CASES,
        config={"top_k": 5},
        metric_names=["recall@5"],
        golden_hash="h",
        judge=FakeJudge(),
    )
    assert record.judged is True
    assert "faithfulness" in record.metrics


def _flaky_cases(n: int) -> list[GoldenCase]:
    # Distinct questions: _Flaky counts attempts per query, and identical questions would
    # share a counter so only the first case would ever see a failure.
    return [
        GoldenCase(
            id=f"q{i}",
            question=f"Gravity is the attraction between masses. (case {i})",
            required_passages=[
                Passage(
                    doc_id="physics",
                    char_start=0,
                    char_end=41,
                    text="Gravity is the attraction between masses.",
                )
            ],
        )
        for i in range(n)
    ]


class _Flaky:
    """Fails the first `failures` attempts on every case, then succeeds."""

    def __init__(self, failures: int):
        self.failures = failures
        self.attempts: dict[str, int] = {}

    def retrieve(self, query, index, config):
        self.attempts[query] = self.attempts.get(query, 0) + 1
        if self.attempts[query] <= self.failures:
            raise TimeoutError("upstream timed out")
        return RetrievalTrace(
            steps=[
                Step(
                    query=query,
                    chunks=[
                        Chunk(
                            text="Gravity is the attraction between masses.",
                            doc_id="physics",
                            char_start=0,
                            char_end=41,
                        )
                    ],
                )
            ]
        )


async def test_without_retries_a_transient_failure_is_an_error():
    record = await run_cases(
        _Flaky(failures=1), _flaky_cases(4), config={}, metric_names=["recall@5"], golden_hash="h"
    )
    assert all(r.status == "error" for r in record.case_results)


async def test_a_transient_failure_recovers_and_is_counted():
    record = await run_cases(
        _Flaky(failures=1),
        _flaky_cases(4),
        config={},
        metric_names=["recall@5"],
        golden_hash="h",
        retries=2,
    )
    assert all(r.status == "ok" for r in record.case_results)
    assert record.valid
    # The recovery is reported, not swallowed: a run that needed three tries per case is
    # telling you something the metric cannot.
    assert record.retried == 4
    assert all(r.attempts == 2 for r in record.case_results)


async def test_retries_are_bounded_and_a_permanent_failure_still_errors():
    adapter = _Flaky(failures=99)
    record = await run_cases(
        adapter, _flaky_cases(2), config={}, metric_names=["recall@5"], golden_hash="h", retries=2
    )
    assert all(r.status == "error" for r in record.case_results)
    assert all(r.attempts == 3 for r in record.case_results)  # 1 try + 2 retries, no more
    assert not record.valid


async def test_a_run_with_no_retries_needed_reports_none():
    record = await run_cases(
        _Flaky(failures=0),
        _flaky_cases(3),
        config={},
        metric_names=["recall@5"],
        golden_hash="h",
        retries=2,
    )
    assert record.retried == 0
    assert all(r.attempts == 1 for r in record.case_results)
