"""Searching the configuration space without evaluating all of it."""

from collections.abc import Sequence
from itertools import product
from typing import Any

from ragci.contract import AdapterSpec

Config = dict[str, Any]


def expand_grid(spec: AdapterSpec, only: Sequence[str] | None = None) -> list[Config]:
    """Every combination of the adapter's declared parameter values."""
    declared = {p.name: p.values for p in spec.index_time_params + spec.query_time_params}

    if only is not None:
        unknown = [name for name in only if name not in declared]
        if unknown:
            raise ValueError(f"not declared by the adapter: {', '.join(unknown)}")
        declared = {name: declared[name] for name in only}

    if not declared:
        return [{}]

    names = list(declared)
    return [dict(zip(names, values, strict=True)) for values in product(*declared.values())]


def index_signature(config: Config, index_keys: Sequence[str]) -> tuple:
    """What identifies the index a configuration needs — query-time values excluded."""
    return tuple(repr(config.get(key)) for key in sorted(index_keys))


def order_by_index_cost(configs: Sequence[Config], index_keys: Sequence[str]) -> list[Config]:
    """Group configurations that share an index so it is built once, not once each.

    On a large corpus, rebuilding is the dominant cost of a sweep — ordering the grid
    is the difference between one reindex per index and one per configuration.
    """
    return sorted(
        configs, key=lambda c: (index_signature(c, index_keys), repr(sorted(c.items())))
    )


def count_rebuilds(configs: Sequence[Config], index_keys: Sequence[str]) -> int:
    """How many index builds this order costs, counting consecutive repeats as one."""
    rebuilds, previous = 0, object()
    for config in configs:
        signature = index_signature(config, index_keys)
        if signature != previous:
            rebuilds += 1
            previous = signature
    return rebuilds
