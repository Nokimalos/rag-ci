"""Compare a run against its baseline and decide whether to block the pull request."""

from typing import Literal

from pydantic import BaseModel

from ragci.runner import RunRecord
from ragci.stats import paired_bootstrap_test

DEFAULT_MIN_EFFECT = 0.02
DEFAULT_ALPHA = 0.05

GateReason = Literal[
    "ok",
    "improved",
    "regression",
    "stale_baseline",
    "invalid_run",
    "incomplete_run",
    "no_baseline",
    "no_pairs",
]


class GateDecision(BaseModel):
    passed: bool
    reason: GateReason
    metric: str
    delta: float | None = None
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_pairs: int = 0
    message: str


def _scored(record: RunRecord, metric: str) -> dict[str, float]:
    return {
        result.case_id: result.scores[metric]
        for result in record.case_results
        if result.status == "ok" and metric in result.scores
    }


def align_scores(
    baseline: RunRecord, candidate: RunRecord, metric: str
) -> tuple[list[float], list[float], list[str]]:
    """Pair the two runs on the cases both actually scored, in a stable order."""
    base, cand = _scored(baseline, metric), _scored(candidate, metric)
    case_ids = sorted(set(base) & set(cand))
    return [base[c] for c in case_ids], [cand[c] for c in case_ids], case_ids


def decide(
    baseline: RunRecord | None,
    candidate: RunRecord,
    *,
    metric: str | None = None,
    min_effect: float = DEFAULT_MIN_EFFECT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> GateDecision:
    metric = metric or candidate.primary_metric

    if not candidate.complete:
        return GateDecision(
            passed=False,
            reason="incomplete_run",
            metric=metric,
            message=(
                "This run stopped before all requested work finished. Incomplete runs cannot "
                "be compared against or establish a baseline."
            ),
        )

    if not candidate.valid:
        return GateDecision(
            passed=False,
            reason="invalid_run",
            metric=metric,
            message=(
                f"{candidate.error_rate:.0%} of cases errored. This run says nothing about "
                "retrieval quality — fix the pipeline, then re-run."
            ),
        )

    if baseline is None:
        return GateDecision(
            passed=True,
            reason="no_baseline",
            metric=metric,
            message="No baseline yet. This run establishes it.",
        )

    if baseline.golden_hash != candidate.golden_hash:
        return GateDecision(
            passed=False,
            reason="stale_baseline",
            metric=metric,
            message=(
                "The golden set changed since the baseline was recorded. Comparing these "
                "two runs would measure the golden set, not the pipeline. Re-record the "
                "baseline with `rag-ci gate --update-baseline`."
            ),
        )

    base_scores, cand_scores, _ = align_scores(baseline, candidate, metric)
    if not base_scores:
        return GateDecision(
            passed=False,
            reason="no_pairs",
            metric=metric,
            message=f"No case was scored on {metric} by both runs; nothing to compare.",
        )

    test = paired_bootstrap_test(base_scores, cand_scores, seed=seed)
    significant = test.p_value < alpha
    material = test.delta < -min_effect

    # Both conditions must hold. Significance alone blocks on 0.3% drops nobody cares
    # about; effect size alone blocks on noise. Together they block on real regressions.
    if significant and material:
        return GateDecision(
            passed=False,
            reason="regression",
            metric=metric,
            delta=test.delta,
            p_value=test.p_value,
            ci_low=test.ci_low,
            ci_high=test.ci_high,
            n_pairs=test.n_pairs,
            message=(
                f"{metric} dropped by {abs(test.delta):.3f} "
                f"(95% CI [{test.ci_low:.3f}, {test.ci_high:.3f}], p={test.p_value:.4f}) "
                f"over {test.n_pairs} paired cases."
            ),
        )

    improved = test.delta > min_effect and test.p_value > 1 - alpha
    return GateDecision(
        passed=True,
        reason="improved" if improved else "ok",
        metric=metric,
        delta=test.delta,
        p_value=test.p_value,
        ci_low=test.ci_low,
        ci_high=test.ci_high,
        n_pairs=test.n_pairs,
        message=(
            f"{metric} moved by {test.delta:+.3f} "
            f"(95% CI [{test.ci_low:.3f}, {test.ci_high:.3f}]) over {test.n_pairs} paired cases."
        ),
    )
