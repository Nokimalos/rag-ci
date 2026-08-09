"""Human and machine renderings of a run."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ragci.baseline import GateDecision
from ragci.runner import RunRecord
from ragci.sweep import SweepOutcome


def render_console(record: RunRecord, console: Console | None = None) -> None:
    console = console or Console()

    primary = record.metrics.get(record.primary_metric)
    scored = primary.n if primary else 0
    table = Table(title=f"rag-ci run  ({scored} scored cases)")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("95% CI", justify="right")

    for name, summary in record.metrics.items():
        marker = " *" if name == record.primary_metric else ""
        table.add_row(
            f"{name}{marker}",
            f"{summary.mean:.3f}",
            f"[{summary.ci_low:.3f}, {summary.ci_high:.3f}]",
        )
    console.print(table)

    console.print(
        f"latency p50 {record.timings.latency_ms_p50:.0f} ms  "
        f"p95 {record.timings.latency_ms_p95:.0f} ms"
    )
    if record.cost_usd_per_query is not None:
        console.print(f"cost {record.cost_usd_per_query:.4f} USD per query")
    if record.tokens_per_query is not None:
        console.print(f"tokens {record.tokens_per_query:.0f} per query")

    if not record.valid:
        console.print(
            f"[bold red]INVALID RUN[/] — {record.error_rate:.0%} of cases errored. "
            "This says nothing about retrieval quality."
        )
    if record.degraded_matching:
        console.print(
            "[yellow]Warning:[/] degraded matching — the adapter returned chunks without "
            "character offsets, so coverage fell back to token overlap."
        )


def save_json(record: RunRecord, path: Path) -> None:
    """Timings are excluded so that two identical runs produce identical bytes."""
    payload = record.model_dump(exclude={"timings"})
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> RunRecord:
    return RunRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))


def render_markdown(record: RunRecord, decision: GateDecision | None = None) -> str:
    """A pull request comment: the verdict first, the numbers under it."""
    lines = ["## rag-ci", ""]

    if decision is not None:
        badge = "✅ **Pass**" if decision.passed else "❌ **Fail**"
        lines += [f"### Gate: {badge}", "", decision.message, ""]
        if decision.delta is not None:
            lines += [
                f"`{decision.metric}` change: **{decision.delta:+.3f}** "
                f"(95% CI [{decision.ci_low:.3f}, {decision.ci_high:.3f}], "
                f"p={decision.p_value:.4f}) over {decision.n_pairs} paired cases.",
                "",
            ]

    lines += ["| metric | mean | 95% CI |", "| --- | ---: | ---: |"]
    for name, summary in record.metrics.items():
        marker = " *" if name == record.primary_metric else ""
        lines.append(
            f"| {name}{marker} | {summary.mean:.3f} "
            f"| [{summary.ci_low:.3f}, {summary.ci_high:.3f}] |"
        )
    lines.append("")

    lines.append(
        f"latency p50 {record.timings.latency_ms_p50:.0f} ms · "
        f"p95 {record.timings.latency_ms_p95:.0f} ms"
    )
    if record.cost_usd_per_query is not None:
        lines.append(f"cost {record.cost_usd_per_query:.4f} USD per query")
    lines.append("")

    if not record.valid:
        lines += [
            f"> ⚠️ **Invalid run** — {record.error_rate:.0%} of cases errored. "
            "This says nothing about retrieval quality.",
            "",
        ]
    if record.degraded_matching:
        lines += [
            "> ⚠️ Degraded matching — the adapter returned chunks without character "
            "offsets, so coverage fell back to token overlap.",
            "",
        ]

    return "\n".join(lines)


def render_sweep(outcome: "SweepOutcome", console: Console | None = None) -> None:
    """Ranked configurations, with the cost of the search stated rather than implied."""
    console = console or Console()

    # Last evaluation per configuration: the deepest rung it survived to.
    best = {repr(e.config): e for e in outcome.evaluations}

    table = Table(title=f"rag-ci sweep  ({len(outcome.rungs)} rungs)")
    table.add_column("configuration")
    table.add_column("score", justify="right")
    table.add_column("cases", justify="right")

    ranked = sorted(best.values(), key=lambda e: (-e.rung, -e.score))
    deepest = max(e.rung for e in outcome.evaluations)
    for evaluation in ranked:
        marker = " ← winner" if evaluation.config == outcome.winner else ""
        params = ", ".join(f"{k}={v}" for k, v in sorted(evaluation.config.items())) or "default"
        # Scores from different rungs come from different case counts. Saying so beats
        # letting a reader conclude the sweep picked the lowest number in the column.
        cases = str(evaluation.n_cases) + ("" if evaluation.rung == deepest else " *")
        table.add_row(f"{params}{marker}", f"{evaluation.score:.3f}", cases)
    console.print(table)

    if any(e.rung != deepest for e in ranked):
        console.print(
            "[dim]* eliminated earlier, on fewer cases — these scores are not "
            "comparable with the winner's.[/]"
        )

    if outcome.arbitrary_elimination:
        console.print(
            "[yellow]Warning:[/] configurations were eliminated while tied with the "
            "survivors, so the cut was decided by tie-break rather than by evidence. "
            "This winner is a draw, not a result — add cases and sweep again."
        )

    spent = sum(e.n_cases for e in outcome.evaluations)
    saved = 100 * (1 - spent / outcome.full_grid_cost) if outcome.full_grid_cost else 0
    console.print(
        f"Searched {outcome.n_configs} configurations in {len(outcome.rungs)} rung(s): "
        f"{spent} case-evaluations instead of {outcome.full_grid_cost} ({saved:.0f}% saved)."
    )
