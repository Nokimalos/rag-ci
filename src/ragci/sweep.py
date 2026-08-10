"""Searching the configuration space without evaluating all of it."""

from collections.abc import Callable, Sequence
from itertools import product
from statistics import fmean
from typing import Any

from pydantic import BaseModel

from ragci.contract import AdapterSpec
from ragci.stats import holm_bonferroni, paired_bootstrap_test

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


class Comparison(BaseModel):
    """The winner measured against one configuration that reached the final rung."""

    against: Config
    advantage: float  # winner minus this configuration, on their shared cases
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool  # after Holm-Bonferroni across the whole family


class SweepOutcome(BaseModel):
    winner: Config
    evaluations: list[SweepEvaluation]
    rungs: list[Rung]
    evaluations_run: int
    n_configs: int
    full_grid_cost: int
    # True when a rung eliminated configurations that were tied with the survivors.
    # The cut was then alphabetical, not evidence-based, and the winner is a draw.
    arbitrary_elimination: bool = False
    # The winner against each of its co-finalists. Empty when it had none to beat.
    comparisons: list[Comparison] = []

    @property
    def winner_is_significant(self) -> bool:
        """Did the winner beat every finalist by more than noise?

        False when any comparison failed, and also when there were none to make —
        a configuration that won by default has not been shown to be better.
        """
        return bool(self.comparisons) and all(c.significant for c in self.comparisons)

    @property
    def contenders(self) -> list[Comparison]:
        """Finalists the winner did not separate itself from."""
        return [c for c in self.comparisons if not c.significant]

    @property
    def decisive(self) -> bool:
        """Is this a result, or a ranking that happened to have a top row?

        A sole configuration is decisive by default — there was nothing to distinguish
        it from. Anything else has to earn it against the field.
        """
        if self.arbitrary_elimination:
            return False
        return self.winner_is_significant if self.comparisons else self.n_configs == 1


def plan_rungs(n_configs: int, n_cases: int, *, eta: int = 3, min_cases: int = 10) -> list[Rung]:
    """Fewer configurations against more cases at each step.

    A bad configuration should be dropped on cheap evidence; only survivors earn a
    full evaluation. Total cost is roughly n_configs x min_cases x rungs rather than
    n_configs x n_cases.

    The floor is two, not one: a rung that narrows to a single configuration leaves the
    final rung with nobody to compare the winner against, and a winner that ran alone
    cannot be shown to be better than anything. Carrying a runner-up to the end costs
    one more full evaluation and is what makes the verdict a verdict.
    """
    configs_left, cases = n_configs, max(min_cases, 1)

    steps = 1
    while configs_left > 2 and cases * eta <= n_cases:
        configs_left = max(2, configs_left // eta)
        cases *= eta
        steps += 1

    rungs: list[Rung] = []
    configs_left, cases = n_configs, max(min_cases, 1)
    for index in range(steps):
        is_last = index == steps - 1
        rungs.append(
            Rung(index=index, n_configs=configs_left, n_cases=n_cases if is_last else cases)
        )
        configs_left = max(2, configs_left // eta)
        cases *= eta
    return rungs


def _cut_is_arbitrary(ranked: list[tuple[float, Config]], keep: int) -> bool:
    """Did the last survivor tie with the first casualty?

    When it did, the cut fell inside a group of equal scores and was settled by the
    tie-break, not by evidence. Reporting a winner from that is reporting a coin toss.
    """
    if keep >= len(ranked):
        return False
    return ranked[keep - 1][0] == ranked[keep][0]


def _rank(scored: list[tuple[float, Config]]) -> list[tuple[float, Config]]:
    # Ties break on the configuration's own ordering so a rerun picks the same winner.
    return sorted(scored, key=lambda pair: (-pair[0], repr(sorted(pair[1].items()))))


def _verdict(
    winner: Config,
    finalists: Sequence[tuple[Config, list[float | None]]],
    *,
    alpha: float,
    seed: int,
) -> list[Comparison]:
    """Comparisons for the winner against the rest of the final rung, if there is a rest."""
    winner_scores = next((s for c, s in finalists if c == winner), [])
    if not any(s is not None for s in winner_scores):
        return []
    others = [(c, s) for c, s in finalists if c != winner]
    return compare_finalists(winner_scores, others, alpha=alpha, seed=seed)


def compare_finalists(
    winner_scores: Sequence[float | None],
    others: Sequence[tuple[Config, Sequence[float | None]]],
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[Comparison]:
    """Test the winner against everything that reached the final rung with it.

    Ranking picks the largest mean. This asks the different question: is that lead
    bigger than the noise? Finalists all ran the same cases in the same order, so the
    comparison is paired — which is the only kind a sweep can make honestly.

    Holm-Bonferroni is not optional here. Comparing a winner against nineteen runners-up
    at alpha=0.05 makes one spurious "significantly better" the expected outcome.
    """
    tests, compared = [], []
    for config, scores in others:
        pairs = [
            (w, o)
            for w, o in zip(winner_scores, scores, strict=True)
            if w is not None and o is not None
        ]
        if not pairs:  # every shared case errored on one side or the other
            continue
        # baseline=winner, candidate=other. The p-value then reads "the other is not
        # worse than the winner", so a small p-value is the winner genuinely ahead.
        tests.append(paired_bootstrap_test([w for w, _ in pairs], [o for _, o in pairs], seed=seed))
        compared.append(config)

    if not tests:
        return []
    verdicts = holm_bonferroni([test.p_value for test in tests], alpha=alpha)

    return [
        Comparison(
            against=config,
            # Signs flip because the test measured the other side's deficit. The
            # `+ 0.0` turns IEEE negative zero back into zero, so an exact tie prints
            # as 0.000 rather than a puzzling -0.000.
            advantage=-test.delta + 0.0,
            ci_low=-test.ci_high + 0.0,
            ci_high=-test.ci_low + 0.0,
            p_value=test.p_value,
            significant=verdict,
        )
        for config, test, verdict in zip(compared, tests, verdicts, strict=True)
    ]


def successive_halving(
    configs: Sequence[Config],
    evaluate: Callable[[Config, int], Sequence[float | None]],
    *,
    n_cases: int,
    eta: int = 3,
    min_cases: int = 10,
    alpha: float = 0.05,
    seed: int = 0,
) -> SweepOutcome:
    """Run the grid through shrinking rungs and return the surviving configuration.

    `evaluate` returns one score per case, not a mean: ranking needs the mean, but
    testing the winner needs the pairs.
    """
    if not configs:
        raise ValueError("cannot sweep: no configurations in the grid")

    rungs = plan_rungs(len(configs), n_cases, eta=eta, min_cases=min_cases)
    survivors = list(configs)
    evaluations: list[SweepEvaluation] = []
    arbitrary = False
    finalists: list[tuple[Config, list[float | None]]] = []

    for rung in rungs:
        scored, finalists = [], []
        for config in survivors:
            scores = list(evaluate(config, rung.n_cases))
            measured = [s for s in scores if s is not None]
            mean = fmean(measured) if measured else 0.0
            evaluations.append(
                SweepEvaluation(config=config, score=mean, n_cases=rung.n_cases, rung=rung.index)
            )
            scored.append((mean, config))
            finalists.append((config, scores))
        ranked = _rank(scored)
        # max(2, ...): the final rung needs a field, not a lone survivor. See plan_rungs.
        keep = max(2, len(ranked) // eta) if rung.index < len(rungs) - 1 else 1
        arbitrary = arbitrary or _cut_is_arbitrary(ranked, keep)
        survivors = [config for _, config in ranked[:keep]]

    return SweepOutcome(
        winner=survivors[0],
        evaluations=evaluations,
        rungs=rungs,
        evaluations_run=len(evaluations),
        n_configs=len(configs),
        full_grid_cost=len(configs) * n_cases,
        arbitrary_elimination=arbitrary,
        comparisons=_verdict(survivors[0], finalists, alpha=alpha, seed=seed),
    )


async def _run_halving(
    configs: Sequence[Config],
    evaluate,
    n_cases: int,
    eta: int,
    min_cases: int,
    index_keys: Sequence[str],
    alpha: float = 0.05,
    seed: int = 0,
) -> SweepOutcome:
    """successive_halving with an awaited evaluator, reordering survivors each rung
    so configurations sharing an index stay adjacent."""
    rungs = plan_rungs(len(configs), n_cases, eta=eta, min_cases=min_cases)
    survivors = list(configs)
    evaluations: list[SweepEvaluation] = []
    arbitrary = False
    finalists: list[tuple[Config, list[float | None]]] = []

    for rung in rungs:
        survivors = order_by_index_cost(survivors, index_keys)
        scored, finalists = [], []
        for config in survivors:
            scores = list(await evaluate(config, rung.n_cases))
            measured = [s for s in scores if s is not None]
            mean = fmean(measured) if measured else 0.0
            evaluations.append(
                SweepEvaluation(config=config, score=mean, n_cases=rung.n_cases, rung=rung.index)
            )
            scored.append((mean, config))
            finalists.append((config, scores))
        ranked = _rank(scored)
        # max(2, ...): the final rung needs a field, not a lone survivor. See plan_rungs.
        keep = max(2, len(ranked) // eta) if rung.index < len(rungs) - 1 else 1
        arbitrary = arbitrary or _cut_is_arbitrary(ranked, keep)
        survivors = [config for _, config in ranked[:keep]]

    return SweepOutcome(
        winner=survivors[0],
        evaluations=evaluations,
        rungs=rungs,
        evaluations_run=len(evaluations),
        n_configs=len(configs),
        full_grid_cost=len(configs) * n_cases,
        arbitrary_elimination=arbitrary,
        comparisons=_verdict(survivors[0], finalists, alpha=alpha, seed=seed),
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
    alpha: float = 0.05,
    corpus: Any = None,
) -> SweepOutcome:
    """Sweep a real adapter, rebuilding its index as rarely as the grid allows."""
    from ragci.runner import _call, run_cases  # local: keeps sweep importable on its own

    cases = list(cases)
    if not cases:
        raise ValueError("cannot sweep: no cases in the golden set")

    grid = expand_grid(spec, only=only)
    index_keys = [p.name for p in spec.index_time_params if only is None or p.name in only]
    total_cases = n_cases or len(cases)
    built: dict[tuple, Any] = {}

    async def evaluate(config: Config, rung_cases: int) -> list[float]:
        index = None
        if hasattr(adapter, "build_index"):
            signature = index_signature(config, index_keys)
            if signature not in built:
                built.clear()  # one live index at a time: they can be large
                built[signature] = await _call(adapter.build_index, corpus, config)
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
        # One slot per case, in case order, so finalists line up pair-for-pair. An
        # errored case is None rather than dropped: dropping it would shorten this
        # configuration's list and silently misalign every later pair.
        return [
            result.scores.get(metric) if result.status == "ok" else None
            for result in record.case_results
        ]

    return await _run_halving(
        grid, evaluate, total_cases, eta, min_cases, index_keys, alpha=alpha, seed=seed
    )
