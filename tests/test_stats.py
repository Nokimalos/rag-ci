import numpy as np
import pytest

from ragci.stats import bootstrap_ci, holm_bonferroni, paired_bootstrap_test


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


def test_identical_samples_have_zero_delta_and_no_significance():
    values = [0.0, 1.0, 1.0, 0.0, 1.0] * 20
    result = paired_bootstrap_test(values, values)
    assert result.delta == pytest.approx(0.0)
    assert result.p_value > 0.05
    assert result.n_pairs == 100


def test_a_clear_regression_is_significant():
    rng = np.random.default_rng(11)
    baseline = rng.uniform(0.6, 1.0, 200).tolist()
    candidate = [value - 0.25 for value in baseline]
    result = paired_bootstrap_test(baseline, candidate)
    assert result.delta == pytest.approx(-0.25, abs=0.01)
    assert result.p_value < 0.01
    assert result.ci_high < 0


def test_a_clear_improvement_is_not_flagged_as_regression():
    rng = np.random.default_rng(12)
    baseline = rng.uniform(0.2, 0.6, 200).tolist()
    candidate = [value + 0.25 for value in baseline]
    result = paired_bootstrap_test(baseline, candidate)
    assert result.delta > 0
    # p_value answers "is this NOT a regression", so an improvement scores high.
    assert result.p_value > 0.95


def test_noise_that_flips_sign_case_to_case_is_not_significant():
    rng = np.random.default_rng(13)
    baseline = rng.uniform(0.0, 1.0, 25).tolist()
    candidate = [value + rng.normal(0.0, 0.05) for value in baseline]
    result = paired_bootstrap_test(baseline, candidate)
    assert result.p_value > 0.05


def test_a_uniform_tiny_drop_is_significant_which_is_why_min_effect_exists():
    # Every case drops by exactly 0.01, so the differences have zero variance and the
    # test is certain. Significance alone would block a pull request over this, which is
    # precisely why the gate also demands the effect be material.
    rng = np.random.default_rng(13)
    baseline = rng.uniform(0.0, 1.0, 25).tolist()
    candidate = [value - 0.01 for value in baseline]
    result = paired_bootstrap_test(baseline, candidate)
    assert result.p_value < 0.05
    assert result.delta == pytest.approx(-0.01)


def test_pairing_detects_a_shift_that_unpaired_comparison_would_miss():
    # Per-case scores vary wildly, but every case drops by the same small amount.
    # Comparing two independent means drowns that shift in between-case variance;
    # pairing removes it entirely. This is why the gate must be paired.
    rng = np.random.default_rng(14)
    baseline = rng.uniform(0.0, 1.0, 150).tolist()
    candidate = [value - 0.05 for value in baseline]

    paired = paired_bootstrap_test(baseline, candidate)
    unpaired_baseline = bootstrap_ci(baseline)
    unpaired_candidate = bootstrap_ci(candidate)

    assert paired.p_value < 0.01
    # The unpaired intervals overlap heavily: the same shift looks like nothing.
    assert unpaired_candidate.ci_high > unpaired_baseline.ci_low


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same number of cases"):
        paired_bootstrap_test([0.1, 0.2], [0.1])


def test_paired_empty_input_is_rejected():
    with pytest.raises(ValueError, match="at least one pair"):
        paired_bootstrap_test([], [])


def test_paired_test_is_reproducible():
    a = [0.1, 0.5, 0.9, 0.3, 0.7]
    b = [0.2, 0.4, 0.8, 0.4, 0.6]
    assert paired_bootstrap_test(a, b, seed=5) == paired_bootstrap_test(a, b, seed=5)


def test_a_single_comparison_is_plain_thresholding():
    assert holm_bonferroni([0.04]) == [True]
    assert holm_bonferroni([0.06]) == [False]


def test_verdicts_come_back_in_input_order():
    assert holm_bonferroni([0.9, 0.001, 0.9]) == [False, True, False]


def test_the_smallest_p_faces_the_strictest_threshold():
    # Three comparisons: the smallest must beat alpha/3 = 0.0167.
    assert holm_bonferroni([0.02, 0.03, 0.04]) == [False, False, False]
    assert holm_bonferroni([0.01, 0.03, 0.04]) == [True, False, False]


def test_holm_is_more_powerful_than_plain_bonferroni():
    # Bonferroni would need every p below 0.05/3; Holm relaxes after each rejection.
    assert holm_bonferroni([0.001, 0.02, 0.9])[:2] == [True, True]


def test_rejection_stops_at_the_first_failure_in_sorted_order():
    # Sorted: 0.001, 0.002, 0.9. The first two clear their thresholds (0.0167, 0.025);
    # 0.9 does not, so nothing ranked above it can be rejected either. Step-down runs
    # over sorted order, not input order — both small values are rejected.
    assert holm_bonferroni([0.001, 0.9, 0.002]) == [True, False, True]


def test_an_empty_family_is_an_empty_verdict():
    assert holm_bonferroni([]) == []


def test_alpha_is_configurable():
    assert holm_bonferroni([0.08], alpha=0.10) == [True]


def test_a_p_value_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        holm_bonferroni([1.5])
