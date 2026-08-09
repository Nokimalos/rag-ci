import pytest

from ragci.contract import AdapterSpec, ParamSpec
from ragci.sweep import (
    count_rebuilds,
    expand_grid,
    index_signature,
    order_by_index_cost,
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
