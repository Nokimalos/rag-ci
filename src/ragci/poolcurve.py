"""Extrapolating a sub-corpus measurement to the full corpus, honestly."""

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

MIN_R_SQUARED = 0.9
BOOTSTRAP_RESAMPLES = 2000


class PoolPoint(BaseModel):
    pool_size: int
    score: float


class PoolCurve(BaseModel):
    slope: float
    intercept: float
    r_squared: float
    reliable: bool
    # Every measured score identical: no trend to fit, and no evidence of degradation.
    flat: bool = False
    extrapolated: float
    ci_low: float
    ci_high: float
    target_pool_size: int


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def fit_pool_curve(
    points: Sequence[PoolPoint],
    *,
    target_pool_size: int,
    min_r_squared: float = MIN_R_SQUARED,
    seed: int = 0,
) -> PoolCurve:
    """Fit score against log10(pool size) and project to the full corpus.

    Retrieval gets harder as distractors accumulate, roughly linearly in the log of
    the pool size. When the observed points do not follow that shape, the honest
    output is "unreliable" — not a confident number nobody should act on.
    """
    if len(points) < 2:
        raise ValueError("fitting a curve needs at least two points")
    if any(p.pool_size <= 0 for p in points) or target_pool_size <= 0:
        raise ValueError("pool sizes must be positive")

    x = np.array([math.log10(p.pool_size) for p in points])
    y = np.array([p.score for p in points])
    slope, intercept = _fit(x, y)

    residual = float(np.sum((y - (slope * x + intercept)) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    # Identical scores at every pool size leave nothing for a line to explain. Calling
    # that a perfect fit would project a flat plateau to any corpus size with a
    # zero-width interval — maximum confidence from an absence of evidence. Not
    # observing degradation across the sizes you measured is not proof there is none
    # three decades further out.
    flat = total == 0.0
    r_squared = 1.0 if flat else 1.0 - residual / total

    target_x = math.log10(target_pool_size)
    extrapolated = float(np.clip(slope * target_x + intercept, 0.0, 1.0))

    # Bootstrap the fit itself for the band: resample points, refit, project.
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(points), size=(BOOTSTRAP_RESAMPLES, len(points)))
    projections = []
    for draw in draws:
        if len(set(draw.tolist())) < 2:
            continue  # a degenerate resample cannot define a line
        s, i = _fit(x[draw], y[draw])
        projections.append(float(np.clip(s * target_x + i, 0.0, 1.0)))
    band = np.quantile(projections, [0.025, 0.975]) if projections else (extrapolated,) * 2

    return PoolCurve(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        flat=flat,
        # Two points always fit a line perfectly; r_squared carries no information there.
        # Neither does a flat series — see above.
        reliable=len(points) >= 3 and not flat and r_squared >= min_r_squared,
        extrapolated=extrapolated,
        ci_low=float(band[0]),
        ci_high=float(band[1]),
        target_pool_size=target_pool_size,
    )
