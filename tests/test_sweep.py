import pytest

from ragci.contract import AdapterSpec, ParamSpec
from ragci.sweep import (
    count_rebuilds,
    expand_grid,
    index_signature,
    order_by_index_cost,
    plan_rungs,
    successive_halving,
)


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
    assert [r.n_configs for r in rungs] == [9, 3, 1]
    assert [r.n_cases for r in rungs] == [10, 30, 90]


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
    outcome = successive_halving(configs, lambda config, n: config["k"] / 10, n_cases=90, eta=3)
    assert outcome.winner == {"k": 8}


def test_halving_costs_less_than_the_full_grid():
    configs = [{"k": i} for i in range(9)]
    outcome = successive_halving(configs, lambda config, n: config["k"] / 10, n_cases=90, eta=3)
    # Full grid would be 9 configs x 90 cases = 810 case-evaluations.
    assert outcome.full_grid_cost == 810
    assert sum(e.n_cases for e in outcome.evaluations) < outcome.full_grid_cost


def test_every_configuration_is_evaluated_at_least_once():
    configs = [{"k": i} for i in range(9)]
    outcome = successive_halving(configs, lambda config, n: 0.5, n_cases=90)
    assert {repr(e.config) for e in outcome.evaluations} >= {repr(c) for c in configs}


def test_an_eliminated_configuration_is_not_evaluated_again():
    configs = [{"k": i} for i in range(9)]
    outcome = successive_halving(configs, lambda config, n: config["k"] / 10, n_cases=90, eta=3)
    # The worst configuration appears only in rung 0.
    worst = [e for e in outcome.evaluations if e.config == {"k": 0}]
    assert [e.rung for e in worst] == [0]


def test_the_evaluator_receives_the_rung_case_count():
    seen: list[int] = []

    def evaluate(config, n_cases):
        seen.append(n_cases)
        return 0.5

    successive_halving([{"k": i} for i in range(9)], evaluate, n_cases=90, eta=3)
    assert set(seen) == {10, 30, 90}


def test_a_tie_resolves_deterministically():
    configs = [{"k": i} for i in range(9)]
    first = successive_halving(configs, lambda config, n: 0.5, n_cases=90)
    second = successive_halving(configs, lambda config, n: 0.5, n_cases=90)
    assert first.winner == second.winner


def test_an_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="no configurations"):
        successive_halving([], lambda config, n: 0.0, n_cases=10)
