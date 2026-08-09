"""Uncertainty quantification. Deliberately free of any retrieval concept.

A bare score is not an answer: recall@10 moving 0.71 -> 0.74 on 200 questions is
indistinguishable from noise, and reporting it without an interval invites the wrong call.
"""

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel


class MetricSummary(BaseModel):
    mean: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MetricSummary:
    """Percentile bootstrap interval for the mean of per-case scores."""
    if len(values) == 0:
        raise ValueError("bootstrap_ci requires at least one value")

    observations = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(observations), size=(n_resamples, len(observations)))
    means = observations[draws].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return MetricSummary(
        mean=float(observations.mean()),
        ci_low=float(low),
        ci_high=float(high),
        n=len(observations),
    )


class PairedTestResult(BaseModel):
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_pairs: int


def paired_bootstrap_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedTestResult:
    """Paired bootstrap over per-case differences.

    Pairing is not a refinement, it is the whole point: per-case scores vary far more
    between questions than between two versions of a pipeline, so an unpaired comparison
    buries a real shift under between-case variance.

    `p_value` is one-sided and answers "is the candidate NOT worse": the fraction of
    resamples whose mean difference is at least zero. A collapsing p_value means a
    regression the data actually supports.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired_bootstrap_test needs the same number of cases on both sides")
    if len(baseline) == 0:
        raise ValueError("paired_bootstrap_test requires at least one pair")

    differences = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    resampled = differences[draws].mean(axis=1)

    low, high = np.quantile(resampled, [0.025, 0.975])
    return PairedTestResult(
        delta=float(differences.mean()),
        ci_low=float(low),
        ci_high=float(high),
        p_value=float((resampled >= 0.0).mean()),
        n_pairs=len(differences),
    )
