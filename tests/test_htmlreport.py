"""The HTML report has one job the terminal cannot do: travel. It must not mislead."""

import re

import pytest

from ragci.htmlreport import render_html
from ragci.poolcurve import PoolCurve
from ragci.sweep import Comparison, Rung, SweepEvaluation, SweepOutcome


def _outcome(**overrides) -> SweepOutcome:
    defaults = dict(
        winner={"top_k": 5},
        evaluations=[
            SweepEvaluation(config={"top_k": 5}, score=0.91, n_cases=90, rung=1),
            SweepEvaluation(config={"top_k": 3}, score=0.84, n_cases=90, rung=1),
            SweepEvaluation(config={"top_k": 1}, score=0.55, n_cases=10, rung=0),
        ],
        rungs=[Rung(index=0, n_configs=3, n_cases=10), Rung(index=1, n_configs=2, n_cases=90)],
        evaluations_run=3,
        n_configs=3,
        full_grid_cost=270,
    )
    return SweepOutcome(**{**defaults, **overrides})


def _comparison(**overrides) -> Comparison:
    defaults = dict(
        against={"top_k": 3},
        advantage=0.07,
        ci_low=0.02,
        ci_high=0.12,
        p_value=0.0031,
        significant=True,
    )
    return Comparison(**{**defaults, **overrides})


def test_the_page_is_self_contained():
    # A report that fetches anything renders as a blank box the day the host moves.
    page = render_html(_outcome())
    assert "<style>" in page
    assert not re.search(r"<script|https?://|src=|@import", page)


def test_the_winner_and_its_score_are_there():
    page = render_html(_outcome())
    assert "top_k=5" in page
    assert "0.910" in page


def test_a_confirmed_winner_says_so_with_its_p_value():
    page = render_html(_outcome(comparisons=[_comparison()]))
    assert "Winner confirmed" in page
    assert "0.0031" in page


def test_an_unseparated_field_is_not_presented_as_a_result():
    # The failure the whole verdict exists to prevent, and a page makes a coin toss look
    # far more convincing than a table does.
    page = render_html(_outcome(comparisons=[_comparison(significant=False, p_value=0.4102)]))
    assert "No clear winner" in page
    assert "0.4102" in page
    assert "Winner confirmed" not in page


def test_a_tie_break_cut_is_stated_on_the_page():
    page = render_html(_outcome(arbitrary_elimination=True))
    assert "tie-break" in page


def test_configurations_from_earlier_rungs_are_separated_from_the_finalists():
    # Charting a 10-case score beside a 90-case one invites a comparison the numbers do
    # not support. They get their own table, with the reason attached.
    page = render_html(_outcome())
    assert "Eliminated earlier" in page
    assert "not comparable" in page
    assert page.index("Final rung") < page.index("Eliminated earlier")


def test_a_sweep_with_no_eliminations_has_no_empty_section():
    outcome = _outcome(
        evaluations=[SweepEvaluation(config={"top_k": 5}, score=0.91, n_cases=90, rung=0)],
        rungs=[Rung(index=0, n_configs=1, n_cases=90)],
    )
    assert "Eliminated earlier" not in render_html(outcome)


def test_a_reliable_projection_is_labelled_as_a_projection():
    curve = PoolCurve(
        slope=-0.1,
        intercept=1.0,
        r_squared=0.97,
        reliable=True,
        extrapolated=0.612,
        ci_low=0.548,
        ci_high=0.671,
        target_pool_size=1_000_000,
    )
    page = render_html(_outcome(pool_curve=curve))
    assert "1,000,000" in page
    assert "not a measurement" in page


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(reliable=False, flat=True, r_squared=1.0), "no trend to extend"),
        (dict(reliable=False, r_squared=0.42), "would invent a number"),
    ],
)
def test_a_projection_that_cannot_be_made_says_why(kwargs, expected):
    curve = PoolCurve(
        slope=-0.1,
        intercept=1.0,
        extrapolated=0.6,
        ci_low=0.6,
        ci_high=0.6,
        target_pool_size=1_000_000,
        **kwargs,
    )
    assert expected in render_html(_outcome(pool_curve=curve))


def test_configuration_values_are_escaped():
    # Parameter values come from the user's adapter and land in the page verbatim.
    page = render_html(
        _outcome(
            winner={"prompt": "<script>alert(1)</script>"},
            evaluations=[
                SweepEvaluation(
                    config={"prompt": "<script>alert(1)</script>"}, score=0.5, n_cases=10, rung=0
                )
            ],
        )
    )
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
