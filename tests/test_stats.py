import numpy as np
import pytest

from ragci.stats import bootstrap_ci


def test_mean_is_the_sample_mean():
    assert bootstrap_ci([0.0, 1.0, 1.0, 0.0]).mean == pytest.approx(0.5)


def test_interval_brackets_the_mean():
    summary = bootstrap_ci([0.2, 0.4, 0.6, 0.8, 1.0])
    assert summary.ci_low <= summary.mean <= summary.ci_high


def test_interval_covers_the_true_mean_most_of_the_time():
    # Nominal 95% coverage: allow a generous band, we are testing correctness not calibration.
    rng = np.random.default_rng(1234)
    true_mean = 0.6
    hits = 0
    trials = 200
    for trial in range(trials):
        sample = rng.binomial(1, true_mean, size=120).astype(float)
        summary = bootstrap_ci(sample.tolist(), n_resamples=2000, seed=trial)
        hits += summary.ci_low <= true_mean <= summary.ci_high
    assert 0.88 <= hits / trials <= 1.0


def test_interval_narrows_as_the_sample_grows():
    rng = np.random.default_rng(7)
    small = bootstrap_ci(rng.binomial(1, 0.5, 30).astype(float).tolist(), n_resamples=2000)
    large = bootstrap_ci(rng.binomial(1, 0.5, 3000).astype(float).tolist(), n_resamples=2000)
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_a_constant_sample_has_a_degenerate_interval():
    summary = bootstrap_ci([0.7] * 50)
    assert summary.ci_low == pytest.approx(0.7)
    assert summary.ci_high == pytest.approx(0.7)


def test_same_seed_gives_identical_intervals():
    values = [0.1, 0.9, 0.4, 0.6, 0.5]
    assert bootstrap_ci(values, seed=42) == bootstrap_ci(values, seed=42)


def test_different_seeds_give_close_but_distinct_intervals():
    # Needs a large continuous sample: over a handful of discrete values the set of
    # attainable resample means is small, so every seed lands on the same quantiles.
    rng = np.random.default_rng(3)
    values = rng.uniform(0.0, 1.0, 200).tolist()
    a = bootstrap_ci(values, n_resamples=500, seed=1)
    b = bootstrap_ci(values, n_resamples=500, seed=2)
    assert a != b
    assert abs(a.ci_low - b.ci_low) < 0.05


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_ci([])
