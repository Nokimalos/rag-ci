"""Searching the configuration space without evaluating all of it."""

from collections.abc import Callable, Sequence
from itertools import product
from typing import Any

from pydantic import BaseModel

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
    return sorted(configs, key=lambda c: (index_signature(c, index_keys), repr(sorted(c.items()))))


def count_rebuilds(configs: Sequence[Config], index_keys: Sequence[str]) -> int:
    """How many index builds this order costs, counting consecutive repeats as one."""
    rebuilds, previous = 0, object()
    for config in configs:
        signature = index_signature(config, index_keys)
        if signature != previous:
            rebuilds += 1
            previous = signature
    return rebuilds


class Rung(BaseModel):
    index: int
    n_configs: int
    n_cases: int


class SweepEvaluation(BaseModel):
    config: Config
    score: float
    n_cases: int
    rung: int


class SweepOutcome(BaseModel):
    winner: Config
    evaluations: list[SweepEvaluation]
    rungs: list[Rung]
    evaluations_run: int
    n_configs: int
    full_grid_cost: int


def plan_rungs(n_configs: int, n_cases: int, *, eta: int = 3, min_cases: int = 10) -> list[Rung]:
    """Fewer configurations against more cases at each step.

    A bad configuration should be dropped on cheap evidence; only survivors earn a
    full evaluation. Total cost is roughly n_configs x min_cases x rungs rather than
    n_configs x n_cases.
    """
    configs_left, cases = n_configs, max(min_cases, 1)

    steps = 1
    while configs_left > 1 and cases * eta <= n_cases:
        configs_left = max(1, configs_left // eta)
        cases *= eta
        steps += 1

    rungs: list[Rung] = []
    configs_left, cases = n_configs, max(min_cases, 1)
    for index in range(steps):
        is_last = index == steps - 1
        rungs.append(
            Rung(index=index, n_configs=configs_left, n_cases=n_cases if is_last else cases)
        )
        configs_left = max(1, configs_left // eta)
        cases *= eta
    return rungs


def _rank(scored: list[tuple[float, Config]]) -> list[tuple[float, Config]]:
    # Ties break on the configuration's own ordering so a rerun picks the same winner.
    return sorted(scored, key=lambda pair: (-pair[0], repr(sorted(pair[1].items()))))


def successive_halving(
    configs: Sequence[Config],
    evaluate: Callable[[Config, int], float],
    *,
    n_cases: int,
    eta: int = 3,
    min_cases: int = 10,
) -> SweepOutcome:
    """Run the grid through shrinking rungs and return the surviving configuration."""
    if not configs:
        raise ValueError("cannot sweep: no configurations in the grid")

    rungs = plan_rungs(len(configs), n_cases, eta=eta, min_cases=min_cases)
    survivors = list(configs)
    evaluations: list[SweepEvaluation] = []

    for rung in rungs:
        scored = []
        for config in survivors:
            score = evaluate(config, rung.n_cases)
            evaluations.append(
                SweepEvaluation(config=config, score=score, n_cases=rung.n_cases, rung=rung.index)
            )
            scored.append((score, config))
        ranked = _rank(scored)
        keep = max(1, len(ranked) // eta) if rung.index < len(rungs) - 1 else 1
        survivors = [config for _, config in ranked[:keep]]

    return SweepOutcome(
        winner=survivors[0],
        evaluations=evaluations,
        rungs=rungs,
        evaluations_run=len(evaluations),
        n_configs=len(configs),
        full_grid_cost=len(configs) * n_cases,
    )


async def _run_halving(
    configs: Sequence[Config],
    evaluate,
    n_cases: int,
    eta: int,
    min_cases: int,
    index_keys: Sequence[str],
) -> SweepOutcome:
    """successive_halving with an awaited evaluator, reordering survivors each rung
    so configurations sharing an index stay adjacent."""
    rungs = plan_rungs(len(configs), n_cases, eta=eta, min_cases=min_cases)
    survivors = list(configs)
    evaluations: list[SweepEvaluation] = []

    for rung in rungs:
        survivors = order_by_index_cost(survivors, index_keys)
        scored = []
        for config in survivors:
            score = await evaluate(config, rung.n_cases)
            evaluations.append(
                SweepEvaluation(config=config, score=score, n_cases=rung.n_cases, rung=rung.index)
            )
            scored.append((score, config))
        ranked = _rank(scored)
        keep = max(1, len(ranked) // eta) if rung.index < len(rungs) - 1 else 1
        survivors = [config for _, config in ranked[:keep]]

    return SweepOutcome(
        winner=survivors[0],
        evaluations=evaluations,
        rungs=rungs,
        evaluations_run=len(evaluations),
        n_configs=len(configs),
        full_grid_cost=len(configs) * n_cases,
    )


async def sweep_adapter(
    adapter,
    cases: Sequence[Any],
    *,
    spec: AdapterSpec,
    metric: str,
    n_cases: int | None = None,
    eta: int = 3,
    min_cases: int = 10,
    only: Sequence[str] | None = None,
    seed: int = 0,
    corpus: Any = None,
) -> SweepOutcome:
    """Sweep a real adapter, rebuilding its index as rarely as the grid allows."""
    from ragci.runner import run_cases  # local: keeps sweep importable on its own

    cases = list(cases)
    if not cases:
        raise ValueError("cannot sweep: no cases in the golden set")

    grid = expand_grid(spec, only=only)
    index_keys = [p.name for p in spec.index_time_params if only is None or p.name in only]
    total_cases = n_cases or len(cases)
    built: dict[tuple, Any] = {}

    async def evaluate(config: Config, rung_cases: int) -> float:
        index = None
        if hasattr(adapter, "build_index"):
            signature = index_signature(config, index_keys)
            if signature not in built:
                built.clear()  # one live index at a time: they can be large
                built[signature] = adapter.build_index(corpus, config)
            index = built[signature]
        record = await run_cases(
            adapter,
            cases[:rung_cases],
            config=config,
            metric_names=[metric],
            golden_hash="sweep",
            index=index,
            seed=seed,
        )
        summary = record.metrics.get(metric)
        return summary.mean if summary else 0.0

    return await _run_halving(grid, evaluate, total_cases, eta, min_cases, index_keys)
