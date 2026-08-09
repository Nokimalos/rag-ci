import pytest

from ragci.baseline import align_scores, decide
from ragci.runner import CaseResult, RunRecord
from ragci.stats import MetricSummary

METRIC = "recall@10"


def _record(scores: dict[str, float], *, golden_hash: str = "h", valid: bool = True) -> RunRecord:
    """Build a record whose per-case scores are exactly `scores`."""
    values = list(scores.values())
    return RunRecord(
        golden_hash=golden_hash,
        config={},
        primary_metric=METRIC,
        metrics={
            METRIC: MetricSummary(
                mean=sum(values) / len(values) if values else 0.0,
                ci_low=0.0,
                ci_high=1.0,
                n=len(values),
            )
        },
        case_results=[
            CaseResult(case_id=case_id, status="ok", scores={METRIC: score})
            for case_id, score in scores.items()
        ],
        valid=valid,
    )


def _spread(offset: float, n: int = 120) -> dict[str, float]:
    return {f"q{i}": min(1.0, max(0.0, (i % 10) / 10 + offset)) for i in range(n)}


def test_no_baseline_passes_and_says_so():
    decision = decide(None, _record(_spread(0.0)))
    assert decision.passed is True
    assert decision.reason == "no_baseline"


def test_an_invalid_run_cannot_be_compared():
    decision = decide(_record(_spread(0.0)), _record(_spread(0.0), valid=False))
    assert decision.passed is False
    assert decision.reason == "invalid_run"


def test_a_changed_golden_set_invalidates_the_baseline():
    decision = decide(
        _record(_spread(0.0), golden_hash="old"),
        _record(_spread(0.0), golden_hash="new"),
    )
    assert decision.passed is False
    assert decision.reason == "stale_baseline"
    assert "golden set" in decision.message.lower()


def test_an_identical_run_passes():
    scores = _spread(0.0)
    decision = decide(_record(scores), _record(scores))
    assert decision.passed is True
    assert decision.reason == "ok"
    assert decision.delta == pytest.approx(0.0)


def test_a_large_regression_blocks():
    decision = decide(_record(_spread(0.3)), _record(_spread(0.0)))
    assert decision.passed is False
    assert decision.reason == "regression"
    assert decision.delta < 0
    assert decision.p_value < 0.05


def test_a_significant_but_negligible_regression_passes():
    # Every case drops by exactly 0.005: statistically undeniable, practically irrelevant.
    baseline_scores = _spread(0.3)
    candidate_scores = {case: score - 0.005 for case, score in baseline_scores.items()}
    decision = decide(_record(baseline_scores), _record(candidate_scores))
    assert decision.p_value < 0.05  # the test does detect it
    assert decision.passed is True  # min_effect keeps it from blocking
    assert decision.reason == "ok"


def test_noise_on_a_small_sample_does_not_block():
    baseline_scores = {"q1": 1.0, "q2": 0.0, "q3": 1.0, "q4": 0.0, "q5": 1.0}
    candidate_scores = {"q1": 0.0, "q2": 0.0, "q3": 1.0, "q4": 0.0, "q5": 1.0}
    decision = decide(_record(baseline_scores), _record(candidate_scores))
    assert decision.passed is True


def test_a_clear_improvement_is_reported_as_such():
    decision = decide(_record(_spread(0.0)), _record(_spread(0.3)))
    assert decision.passed is True
    assert decision.reason == "improved"
    assert decision.delta > 0


def test_only_cases_present_in_both_runs_are_paired():
    baseline_scores = _spread(0.3)
    candidate_scores = dict(list(baseline_scores.items())[:100])  # 20 cases errored out
    decision = decide(_record(baseline_scores), _record(candidate_scores))
    assert decision.n_pairs == 100


def test_alignment_is_order_independent():
    baseline = _record({"q1": 1.0, "q2": 0.0})
    candidate = _record({"q2": 0.0, "q1": 1.0})
    base_scores, cand_scores, case_ids = align_scores(baseline, candidate, METRIC)
    assert case_ids == ["q1", "q2"]
    assert base_scores == [1.0, 0.0]
    assert cand_scores == [1.0, 0.0]


def test_no_overlapping_cases_cannot_be_compared():
    decision = decide(_record({"a": 1.0}), _record({"b": 1.0}))
    assert decision.passed is False
    assert decision.reason == "no_pairs"


def test_errored_cases_are_not_paired():
    baseline = _record({"q1": 1.0, "q2": 1.0})
    candidate = _record({"q1": 1.0, "q2": 1.0})
    candidate.case_results[1].status = "error"
    candidate.case_results[1].scores = {}
    _, _, case_ids = align_scores(baseline, candidate, METRIC)
    assert case_ids == ["q1"]
