"""Human and machine renderings of a run."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ragci.baseline import GateDecision
from ragci.runner import RunRecord


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
