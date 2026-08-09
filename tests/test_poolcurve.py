import math

import pytest

from ragci.poolcurve import PoolPoint, fit_pool_curve


def _log_linear(sizes, intercept, slope):
    return [PoolPoint(pool_size=s, score=intercept + slope * math.log10(s)) for s in sizes]


def test_a_perfect_log_linear_fit_extrapolates_exactly():
    points = _log_linear([1_000, 10_000, 100_000], intercept=1.0, slope=-0.05)
    curve = fit_pool_curve(points, target_pool_size=1_000_000)
    assert curve.r_squared == pytest.approx(1.0)
    assert curve.extrapolated == pytest.approx(1.0 - 0.05 * 6, abs=1e-6)


def test_a_clean_fit_is_reported_as_reliable():
    points = _log_linear([1_000, 10_000, 100_000], intercept=0.9, slope=-0.04)
    assert fit_pool_curve(points, target_pool_size=5_000_000).reliable is True


def test_recall_degrades_as_the_pool_grows():
    points = _log_linear([1_000, 10_000, 100_000], intercept=0.9, slope=-0.04)
    curve = fit_pool_curve(points, target_pool_size=1_000_000)
    assert curve.slope < 0
    assert curve.extrapolated < points[-1].score


def test_a_poor_fit_refuses_to_extrapolate():
    # Deliberately non-log-linear: a confident number here would be a lie.
    points = [
        PoolPoint(pool_size=1_000, score=0.9),
        PoolPoint(pool_size=10_000, score=0.2),
        PoolPoint(pool_size=100_000, score=0.85),
    ]
    curve = fit_pool_curve(points, target_pool_size=1_000_000)
    assert curve.reliable is False
    assert curve.r_squared < 0.9


def test_an_unreliable_curve_still_reports_its_numbers():
    points = [
        PoolPoint(pool_size=1_000, score=0.9),
        PoolPoint(pool_size=10_000, score=0.2),
        PoolPoint(pool_size=100_000, score=0.85),
    ]
    curve = fit_pool_curve(points, target_pool_size=1_000_000)
    assert curve.extrapolated is not None  # reported, but flagged unreliable


def test_the_uncertainty_band_brackets_the_estimate():
    points = _log_linear([1_000, 10_000, 100_000], intercept=0.9, slope=-0.04)
    curve = fit_pool_curve(points, target_pool_size=1_000_000)
    assert curve.ci_low <= curve.extrapolated <= curve.ci_high


def test_a_noisier_fit_widens_the_band():
    clean = fit_pool_curve(
        _log_linear([1_000, 10_000, 100_000, 1_000_000], 0.9, -0.04),
        target_pool_size=10_000_000,
    )
    noisy_points = _log_linear([1_000, 10_000, 100_000, 1_000_000], 0.9, -0.04)
    noisy_points[1].score += 0.08
    noisy_points[2].score -= 0.08
    noisy = fit_pool_curve(noisy_points, target_pool_size=10_000_000)
    assert (noisy.ci_high - noisy.ci_low) > (clean.ci_high - clean.ci_low)


def test_extrapolation_is_clamped_to_a_valid_score():
    points = _log_linear([1_000, 10_000], intercept=0.2, slope=-0.15)
    curve = fit_pool_curve(points, target_pool_size=10_000_000_000)
    assert 0.0 <= curve.extrapolated <= 1.0


def test_fewer_than_three_points_cannot_be_judged_reliable():
    # Two points fit a line perfectly by construction; r_squared says nothing.
    curve = fit_pool_curve(_log_linear([1_000, 10_000], 0.9, -0.04), target_pool_size=100_000)
    assert curve.reliable is False


def test_at_least_two_points_are_required():
    with pytest.raises(ValueError, match="at least two"):
        fit_pool_curve([PoolPoint(pool_size=1_000, score=0.9)], target_pool_size=10_000)


def test_a_non_positive_pool_size_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        fit_pool_curve(
            [PoolPoint(pool_size=0, score=0.9), PoolPoint(pool_size=10, score=0.8)],
            target_pool_size=100,
        )
