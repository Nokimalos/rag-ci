import pytest

from ragci.contract import AdapterSpec, ParamSpec
from ragci.corpus import Document
from ragci.golden import GoldenCase, Passage
from ragci.sweep import (
    count_rebuilds,
    expand_grid,
    index_signature,
    order_by_index_cost,
    plan_rungs,
    successive_halving,
    sweep_adapter,
)
from tests.fixtures.reference_adapter import ReferenceRag


def _spec(index=None, query=None) -> AdapterSpec:
    return AdapterSpec(
        index_time_params=[ParamSpec(name=k, values=v) for k, v in (index or {}).items()],
        query_time_params=[ParamSpec(name=k, values=v) for k, v in (query or {}).items()],
    )


def test_the_grid_is_the_cartesian_product():
    grid = expand_grid(_spec(index={"chunk_size": [256, 512]}, query={"top_k": [5, 10, 20]}))
    assert len(grid) == 6
    assert {"chunk_size": 256, "top_k": 5} in grid


def test_an_adapter_with_no_parameters_yields_one_empty_config():
    assert expand_grid(_spec()) == [{}]


def test_the_grid_is_deterministic():
    spec = _spec(index={"a": [1, 2]}, query={"b": ["x", "y"]})
    assert expand_grid(spec) == expand_grid(spec)


def test_only_restricts_the_grid_to_named_parameters():
    grid = expand_grid(
        _spec(index={"chunk_size": [256, 512]}, query={"top_k": [5, 10]}), only=["top_k"]
    )
    assert len(grid) == 2
    assert all("chunk_size" not in config for config in grid)


def test_only_rejects_an_unknown_parameter():
    with pytest.raises(ValueError, match="not declared"):
        expand_grid(_spec(query={"top_k": [5]}), only=["nonexistent"])


def test_index_signature_ignores_query_time_parameters():
    a = {"chunk_size": 256, "top_k": 5}
    b = {"chunk_size": 256, "top_k": 20}
    assert index_signature(a, ["chunk_size"]) == index_signature(b, ["chunk_size"])


def test_ordering_groups_configurations_sharing_an_index():
    configs = [
        {"chunk_size": 256, "top_k": 5},
        {"chunk_size": 512, "top_k": 5},
        {"chunk_size": 256, "top_k": 10},
        {"chunk_size": 512, "top_k": 10},
    ]
    ordered = order_by_index_cost(configs, ["chunk_size"])
    sizes = [c["chunk_size"] for c in ordered]
    assert sizes == sorted(sizes)  # no interleaving


def test_ordering_minimises_rebuilds():
    configs = [
        {"chunk_size": 256, "top_k": 5},
        {"chunk_size": 512, "top_k": 5},
        {"chunk_size": 256, "top_k": 10},
        {"chunk_size": 512, "top_k": 10},
    ]
    # Worst case is one rebuild per configuration; grouping needs only one per index.
    assert count_rebuilds(configs, ["chunk_size"]) == 4
    assert count_rebuilds(order_by_index_cost(configs, ["chunk_size"]), ["chunk_size"]) == 2


def test_ordering_preserves_the_configuration_set():
    configs = [{"a": 1, "b": 1}, {"a": 2, "b": 1}, {"a": 1, "b": 2}]
    ordered = order_by_index_cost(configs, ["a"])
    assert sorted(map(str, ordered)) == sorted(map(str, configs))


def test_query_only_sweeps_never_rebuild_more_than_once():
    configs = [{"top_k": k} for k in (5, 10, 20)]
    assert count_rebuilds(order_by_index_cost(configs, []), []) == 1


def test_rungs_shrink_configurations_and_grow_cases():
    rungs = plan_rungs(9, 90, eta=3)
    # The final rung holds two, not one. A lone survivor has nothing to be compared
    # against, and a winner nobody ran against cannot be shown to have won.
    assert [r.n_configs for r in rungs] == [9, 3, 2]
    assert [r.n_cases for r in rungs] == [10, 30, 90]


def test_two_configurations_reach_the_final_rung_even_from_a_wide_grid():
    assert plan_rungs(81, 810, eta=3)[-1].n_configs == 2
    assert plan_rungs(2, 90, eta=3)[-1].n_configs == 2


def test_a_single_configuration_gets_a_single_rung():
    rungs = plan_rungs(1, 90, eta=3)
    assert [(r.n_configs, r.n_cases) for r in rungs] == [(1, 90)]


def test_the_last_rung_uses_every_case():
    assert plan_rungs(27, 100, eta=3)[-1].n_cases == 100


def test_a_single_configuration_needs_one_rung():
    rungs = plan_rungs(1, 50)
    assert len(rungs) == 1
    assert rungs[0].n_cases == 50


def test_rungs_never_go_below_the_minimum_case_count():
    assert all(r.n_cases >= 10 for r in plan_rungs(81, 12, eta=3, min_cases=10))


def test_the_best_configuration_wins():
    configs = [{"k": i} for i in range(9)]
    outcome = successive_halving(
        configs, lambda config, n: [config["k"] / 10] * n, n_cases=90, eta=3
    )
    assert outcome.winner == {"k": 8}


def test_halving_costs_less_than_the_full_grid():
    configs = [{"k": i} for i in range(9)]
    outcome = successive_halving(
        configs, lambda config, n: [config["k"] / 10] * n, n_cases=90, eta=3
    )
    # Full grid would be 9 configs x 90 cases = 810 case-evaluations.
    assert outcome.full_grid_cost == 810
    assert sum(e.n_cases for e in outcome.evaluations) < outcome.full_grid_cost


def test_every_configuration_is_evaluated_at_least_once():
    configs = [{"k": i} for i in range(9)]
    outcome = successive_halving(configs, lambda config, n: [0.5] * n, n_cases=90)
    assert {repr(e.config) for e in outcome.evaluations} >= {repr(c) for c in configs}


def test_an_eliminated_configuration_is_not_evaluated_again():
    configs = [{"k": i} for i in range(9)]
    outcome = successive_halving(
        configs, lambda config, n: [config["k"] / 10] * n, n_cases=90, eta=3
    )
    # The worst configuration appears only in rung 0.
    worst = [e for e in outcome.evaluations if e.config == {"k": 0}]
    assert [e.rung for e in worst] == [0]


def test_the_evaluator_receives_the_rung_case_count():
    seen: list[int] = []

    def evaluate(config, n_cases):
        seen.append(n_cases)
        return [0.5] * n_cases

    successive_halving([{"k": i} for i in range(9)], evaluate, n_cases=90, eta=3)
    assert set(seen) == {10, 30, 90}


def test_a_tie_resolves_deterministically():
    configs = [{"k": i} for i in range(9)]
    first = successive_halving(configs, lambda config, n: [0.5] * n, n_cases=90)
    second = successive_halving(configs, lambda config, n: [0.5] * n, n_cases=90)
    assert first.winner == second.winner


def test_an_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="no configurations"):
        successive_halving([], lambda config, n: [0.0] * n, n_cases=10)


def _cases(n: int = 12) -> list[GoldenCase]:
    return [
        GoldenCase(
            id=f"q{i}",
            question="Gravity is the attraction between masses",
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


async def test_a_sweep_returns_a_configuration_from_the_grid():
    outcome = await sweep_adapter(
        ReferenceRag(), _cases(), spec=ReferenceRag.__ragci_spec__, metric="recall@3"
    )
    assert set(outcome.winner) <= {"chunk_size", "top_k"}


async def test_the_sweep_costs_less_than_the_full_grid():
    outcome = await sweep_adapter(
        ReferenceRag(), _cases(36), spec=ReferenceRag.__ragci_spec__, metric="recall@3"
    )
    actual = sum(e.n_cases for e in outcome.evaluations)
    assert outcome.full_grid_cost == 6 * 36  # 2 chunk sizes x 3 top_k x 36 cases
    assert actual < outcome.full_grid_cost


async def test_too_few_cases_degenerates_to_a_full_grid_search():
    # With 12 cases and a 10-case floor there is no room for a second rung, so every
    # configuration is evaluated on everything. That is the correct outcome: you cannot
    # eliminate a configuration on evidence you do not have.
    outcome = await sweep_adapter(
        ReferenceRag(), _cases(12), spec=ReferenceRag.__ragci_spec__, metric="recall@3"
    )
    assert len(outcome.rungs) == 1
    assert sum(e.n_cases for e in outcome.evaluations) == outcome.full_grid_cost


async def test_build_index_is_called_once_per_distinct_index_signature():
    class Counting:
        __ragci_spec__ = ReferenceRag.__ragci_spec__

        def __init__(self):
            self.builds: list[int] = []
            self._inner = ReferenceRag()

        def build_index(self, corpus, config):
            self.builds.append(config.get("chunk_size"))
            return object()

        def retrieve(self, query, index, config):
            return self._inner.retrieve(query, index, config)

    adapter = Counting()
    await sweep_adapter(
        adapter, _cases(), spec=Counting.__ragci_spec__, metric="recall@3", min_cases=12
    )
    # Two chunk sizes, so at most two builds per rung - never one per configuration.
    assert len(adapter.builds) <= 2 * 3


async def test_an_adapter_without_build_index_sweeps_query_parameters_only():
    outcome = await sweep_adapter(
        ReferenceRag(),
        _cases(),
        spec=ReferenceRag.__ragci_spec__,
        metric="recall@3",
        only=["top_k"],
    )
    assert set(outcome.winner) == {"top_k"}


async def test_the_sweep_is_reproducible():
    kwargs = dict(spec=ReferenceRag.__ragci_spec__, metric="recall@3", seed=7)
    first = await sweep_adapter(ReferenceRag(), _cases(), **kwargs)
    second = await sweep_adapter(ReferenceRag(), _cases(), **kwargs)
    assert first.winner == second.winner


async def test_sweeping_with_no_cases_is_rejected():
    with pytest.raises(ValueError, match="no cases"):
        await sweep_adapter(ReferenceRag(), [], spec=ReferenceRag.__ragci_spec__, metric="recall@3")


def test_the_outcome_records_the_grid_size_directly():
    # Deriving it from full_grid_cost / n_cases breaks the moment a rung uses a
    # different case count than the divisor assumes.
    outcome = successive_halving(
        [{"k": i} for i in range(9)], lambda config, n: [0.5] * n, n_cases=90
    )
    assert outcome.n_configs == 9


def test_a_tied_first_rung_is_reported_as_arbitrary():
    # Every configuration scoring the same means the cut was settled by the tie-break,
    # not by evidence. Presenting a "winner" from that is presenting a coin toss.
    outcome = successive_halving(
        [{"k": i} for i in range(9)], lambda config, n: [1.0] * n, n_cases=90
    )
    assert outcome.arbitrary_elimination is True
    assert outcome.decisive is False


def test_a_clear_ranking_is_not_arbitrary():
    outcome = successive_halving(
        [{"k": i} for i in range(9)], lambda config, n: [config["k"] / 10] * n, n_cases=90
    )
    assert outcome.arbitrary_elimination is False
    assert outcome.decisive is True


def test_a_tie_away_from_the_cut_is_not_arbitrary():
    # Ties among configurations that all lose do not affect the cut. The survivors must
    # be distinct, though — three tied finalists means the *final* pick is a coin toss.
    scores = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1}
    outcome = successive_halving(
        [{"k": i} for i in range(9)], lambda config, n: [scores[config["k"]]] * n, n_cases=90
    )
    assert outcome.arbitrary_elimination is False


async def test_an_async_build_index_is_awaited():
    class AsyncIndexing:
        __ragci_spec__ = ReferenceRag.__ragci_spec__

        def __init__(self):
            self._inner = ReferenceRag()
            self.builds = 0

        async def build_index(self, corpus, config):
            self.builds += 1
            return object()

        async def retrieve(self, query, index, config):
            return self._inner.retrieve(query, index, config)

    adapter = AsyncIndexing()
    outcome = await sweep_adapter(
        adapter, _cases(), spec=AsyncIndexing.__ragci_spec__, metric="recall@3"
    )
    assert adapter.builds > 0
    assert set(outcome.winner) <= {"chunk_size", "top_k"}


# --- The winner has to beat the field, not just top the column -----------------------


def _split(wins: int, losses: int, *, winner: bool) -> list[float]:
    """Scores for a configuration that takes `wins` cases and drops `losses`.

    Explicit lists rather than a smooth offset: the paired bootstrap resamples per-case
    differences, so a difference that never changes sign is degenerate — it has zero
    variance and clears any threshold, however small the effect.
    """
    return [1.0] * wins + [0.0] * losses if winner else [0.0] * wins + [1.0] * losses


def _finalists(wins: int, losses: int):
    n = wins + losses
    scores = {0: _split(wins, losses, winner=True), 1: _split(wins, losses, winner=False)}
    return successive_halving([{"k": 0}, {"k": 1}], lambda c, _: scores[c["k"]], n_cases=n)


def test_a_clear_winner_is_significant_against_the_runner_up():
    outcome = _finalists(wins=36, losses=4)  # 0.90 against 0.10
    assert outcome.winner == {"k": 0}
    assert len(outcome.comparisons) == 1

    comparison = outcome.comparisons[0]
    assert comparison.against == {"k": 1}
    assert comparison.advantage == pytest.approx(0.8)
    assert comparison.ci_low > 0  # the interval clears zero
    assert comparison.significant is True
    assert outcome.winner_is_significant is True
    assert outcome.decisive is True


def test_configurations_that_only_differ_by_noise_produce_no_winner():
    # The failure this whole feature exists to prevent. 21 wins against 19 losses is a
    # 0.05 lead that a coin could have produced; ranking alone calls it a result.
    outcome = _finalists(wins=21, losses=19)
    assert outcome.comparisons[0].advantage == pytest.approx(0.05)
    assert outcome.comparisons[0].significant is False
    assert outcome.winner_is_significant is False
    assert outcome.decisive is False
    assert outcome.contenders == outcome.comparisons


def test_the_advantage_is_reported_from_the_winners_side():
    # paired_bootstrap_test measures the runner-up's deficit, so the sign is flipped on
    # the way out. Getting that wrong reports every winner as losing by what it won by.
    comparison = _finalists(wins=30, losses=10).comparisons[0]
    assert comparison.advantage > 0
    assert comparison.ci_low <= comparison.advantage <= comparison.ci_high


def test_a_lone_configuration_has_nothing_to_compare_against():
    outcome = successive_halving([{"k": 0}], lambda c, n: [0.8] * n, n_cases=60)
    assert outcome.comparisons == []
    assert outcome.winner_is_significant is False
    # ...but it is still the answer to "which of these should I use", so not a failure.
    assert outcome.decisive is True


def test_the_correction_never_admits_more_than_the_raw_comparison():
    from ragci.sweep import compare_finalists

    winner = _split(28, 12, winner=True)
    others = [
        ({"k": 1}, _split(28, 12, winner=False)),  # a wide margin
        ({"k": 2}, _split(23, 17, winner=False)),  # a modest one
        ({"k": 3}, _split(21, 19, winner=False)),  # noise
    ]

    corrected = compare_finalists(winner, others, alpha=0.05)
    assert len(corrected) == 3
    # Holm is a step-down procedure: it can only ever be stricter than comparing each
    # p-value to alpha on its own, never looser.
    assert all(c.p_value <= 0.05 or not c.significant for c in corrected)
    assert corrected[-1].significant is False  # noise stays noise


def test_an_errored_case_drops_the_pair_instead_of_misaligning_the_rest():
    # One configuration erroring on case 3 must not shift every later pair by one.
    from ragci.sweep import compare_finalists

    winner = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    other = [0.5, 0.5, None, 0.5, 0.5, 0.5]

    assert compare_finalists(winner, [({"k": 1}, other)])[0].advantage == pytest.approx(0.4)


def test_a_configuration_that_errored_on_every_case_is_skipped_not_crashed():
    from ragci.sweep import compare_finalists

    assert compare_finalists([0.9, 0.8], [({"k": 1}, [None, None])]) == []


# --- Extrapolating past the measured corpus -------------------------------------------


def _docs(n: int, prefix: str = "d"):
    return [Document(doc_id=f"{prefix}{i}", text=f"text {i}") for i in range(n)]


def test_the_split_keeps_every_document_the_golden_set_needs():
    from ragci.sweep import split_pool

    required, distractors = split_pool(_docs(5), {"d1", "d3"})
    assert [d.doc_id for d in required] == ["d1", "d3"]
    assert [d.doc_id for d in distractors] == ["d0", "d2", "d4"]


def test_every_planned_pool_holds_the_required_documents():
    # Slicing the corpus instead would drop the documents the questions are about, and
    # the curve would measure "the answer is gone", not "the answer is harder to find".
    from ragci.sweep import pool_plan

    assert min(pool_plan(4, 996)) >= 4


def test_the_plan_ends_at_the_whole_corpus():
    from ragci.sweep import pool_plan

    assert pool_plan(4, 996)[-1] == 1000
    assert pool_plan(10, 0) == [10]


def test_the_plan_is_log_spaced_and_deduplicated():
    from ragci.sweep import pool_plan

    sizes = pool_plan(10, 9990, points=4)
    assert sizes == sorted(set(sizes))
    assert len(sizes) == 4
    # Log spacing: each step multiplies rather than adds.
    ratios = [b / a for a, b in zip(sizes, sizes[1:], strict=False)]
    assert max(ratios) / min(ratios) < 1.5


def test_a_plan_needs_at_least_two_points():
    from ragci.sweep import pool_plan

    with pytest.raises(ValueError, match="at least two points"):
        pool_plan(4, 996, points=1)


async def test_extrapolation_measures_growing_pools_and_projects():
    from ragci.sweep import measure_pool_curve

    seen: list[int] = []

    class Counting(ReferenceRag):
        async def build_index(self, corpus, config):
            seen.append(len(corpus))
            return None

    curve = await measure_pool_curve(
        Counting(),
        _cases(4),
        config={"chunk_size": 70, "top_k": 3},
        metric="recall@3",
        documents=[*_docs(20), Document(doc_id="physics", text="Gravity.")],
        target_pool_size=1_000_000,
        points=3,
    )
    assert seen == sorted(seen) and len(seen) == 3  # pools grow, one build each
    assert seen[-1] == 21  # the last pool is the whole corpus
    assert curve.target_pool_size == 1_000_000


async def test_extrapolation_refuses_a_corpus_the_questions_are_not_about():
    from ragci.sweep import measure_pool_curve

    class Indexing(ReferenceRag):
        async def build_index(self, corpus, config):
            return None

    with pytest.raises(ValueError, match="not about"):
        await measure_pool_curve(
            Indexing(),
            _cases(2),
            config={},
            metric="recall@3",
            documents=_docs(5),  # no doc_id the golden set references
            target_pool_size=1000,
        )


async def test_extrapolation_needs_build_index():
    from ragci.sweep import measure_pool_curve

    class NoIndex:
        __ragci_spec__ = ReferenceRag.__ragci_spec__

        def retrieve(self, query, index, config):
            return ReferenceRag().retrieve(query, index, config)

    with pytest.raises(ValueError, match="needs build_index"):
        await measure_pool_curve(
            NoIndex(),
            _cases(2),
            config={},
            metric="recall@3",
            documents=_docs(5),
            target_pool_size=1000,
        )


async def test_a_sweep_without_extrapolate_to_reports_no_curve():
    outcome = await sweep_adapter(
        ReferenceRag(), _cases(), spec=ReferenceRag.__ragci_spec__, metric="recall@3"
    )
    assert outcome.pool_curve is None


async def test_extrapolating_without_a_corpus_is_rejected():
    with pytest.raises(ValueError, match="needs a corpus"):
        await sweep_adapter(
            ReferenceRag(),
            _cases(),
            spec=ReferenceRag.__ragci_spec__,
            metric="recall@3",
            extrapolate_to=1000,
        )


# --- Reusing evaluations instead of paying twice --------------------------------------


async def test_a_cached_sweep_skips_the_index_rebuild_too():
    # The index build is the dominant cost on a real corpus. A cache hit that still
    # reindexed would save the cheap half of the work and pay for the expensive half.
    import tempfile

    from ragci import __version__
    from ragci.cache import RunCache

    builds = {"count": 0}

    class Counting(ReferenceRag):
        def build_index(self, corpus, config):
            builds["count"] += 1
            return None

    with tempfile.TemporaryDirectory() as root:
        cache = RunCache(root, fingerprint="fixed", version=__version__)
        kwargs = dict(spec=ReferenceRag.__ragci_spec__, metric="recall@3", cache=cache)

        await sweep_adapter(Counting(), _cases(12), **kwargs)
        first_builds, first_misses = builds["count"], cache.stats.misses
        assert first_builds > 0 and first_misses > 0

        await sweep_adapter(Counting(), _cases(12), **kwargs)
        assert builds["count"] == first_builds, "second sweep rebuilt an index it had cached"
        assert cache.stats.hits >= first_misses


async def test_a_sweep_without_a_cache_behaves_exactly_as_before():
    import tempfile

    from ragci import __version__
    from ragci.cache import RunCache

    plain = await sweep_adapter(
        ReferenceRag(), _cases(12), spec=ReferenceRag.__ragci_spec__, metric="recall@3"
    )
    with tempfile.TemporaryDirectory() as root:
        cached = await sweep_adapter(
            ReferenceRag(),
            _cases(12),
            spec=ReferenceRag.__ragci_spec__,
            metric="recall@3",
            cache=RunCache(root, fingerprint="fixed", version=__version__),
        )
    assert plain.winner == cached.winner
    assert [e.score for e in plain.evaluations] == [e.score for e in cached.evaluations]


# --- Choosing on one set, judging on another ------------------------------------------


def test_no_holdout_leaves_every_case_to_the_search():
    from ragci.sweep import split_holdout

    searchable, reserved = split_holdout(list(range(100)), 0.0)
    assert len(searchable) == 100 and reserved == []


def test_a_holdout_takes_the_tail_and_leaves_the_rest():
    from ragci.sweep import split_holdout

    searchable, reserved = split_holdout(list(range(200)), 0.25)
    assert len(searchable) == 150 and len(reserved) == 50
    assert searchable + reserved == list(range(200))  # nothing lost, nothing duplicated


def test_a_holdout_too_small_to_conclude_anything_is_refused():
    # Reserving ten cases would not remove the bias, it would trade a slightly optimistic
    # verdict for a permanently silent one — and cost a tenth of the search to do it.
    from ragci.sweep import split_holdout

    with pytest.raises(ValueError, match="too few to separate"):
        split_holdout(list(range(100)), 0.1)


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_a_nonsensical_fraction_is_rejected(fraction):
    from ragci.sweep import split_holdout

    with pytest.raises(ValueError, match="between 0 and 1"):
        split_holdout(list(range(100)), fraction)


async def test_the_winner_is_tested_on_cases_the_search_never_saw():
    seen: list[int] = []

    class Watching(ReferenceRag):
        def retrieve(self, query, index, config):
            seen.append(int(query.rsplit(" ", 1)[-1]))
            return super().retrieve(query, index, config)

    cases = [
        GoldenCase(
            id=f"q{i}",
            question=f"Gravity is the attraction between masses {i}",
            required_passages=[
                Passage(
                    doc_id="physics",
                    char_start=0,
                    char_end=41,
                    text="Gravity is the attraction between masses.",
                )
            ],
        )
        for i in range(120)
    ]

    outcome = await sweep_adapter(
        Watching(), cases, spec=ReferenceRag.__ragci_spec__, metric="recall@3", holdout=0.25
    )

    assert outcome.holdout_cases == 30
    assert outcome.comparisons, "the finalists should have been compared on the holdout"
    # The reserved tail is the last 30 questions; the search must have run on the rest.
    assert max(seen) >= 119  # the holdout was evaluated
    assert len([c for c in outcome.evaluations if c.n_cases > 90]) == 0


async def test_a_sweep_without_a_holdout_reports_zero_held_out():
    outcome = await sweep_adapter(
        ReferenceRag(), _cases(12), spec=ReferenceRag.__ragci_spec__, metric="recall@3"
    )
    assert outcome.holdout_cases == 0
