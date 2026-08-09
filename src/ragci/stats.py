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
