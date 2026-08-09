"""Command line entry point."""

import asyncio
import importlib.util
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console

from ragci.baseline import DEFAULT_ALPHA, DEFAULT_MIN_EFFECT, decide
from ragci.contract import AdapterSpec
from ragci.golden import golden_hash, load_golden
from ragci.report import load_json, render_console, render_markdown, save_json
from ragci.runner import run_cases

app = typer.Typer(add_completion=False, help="Regression testing for RAG pipelines.")
console = Console()

TEMPLATE = Path(__file__).parent / "templates" / "adapter_template.py.txt"


def load_adapter(path: Path):
    """Import a module by path and instantiate the single @adapter-decorated class."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)

    candidates = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and isinstance(getattr(value, "__ragci_spec__", None), AdapterSpec)
    ]
    if not candidates:
        console.print(f"[red]Error:[/] no @adapter class found in {path}")
        raise typer.Exit(code=1)
    return candidates[0]()


@app.command()
def init(path: Path = typer.Option(Path("."), help="Where to write ragci_adapter.py")) -> None:
    """Scaffold an adapter file."""
    target = Path(path) / "ragci_adapter.py"
    if target.exists():
        console.print(f"[red]Error:[/] {target} already exists")
        raise typer.Exit(code=1)
    target.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    console.print(f"Wrote {target}. Implement retrieve(), then run: rag-ci run")


@app.command()
def run(
    adapter: Path = typer.Option(Path("ragci_adapter.py"), help="Path to your adapter"),
    golden: Path = typer.Option(Path("golden.jsonl"), help="Path to the golden set"),
    metric: list[str] = typer.Option(None, help="Metrics to compute, first one is primary"),
    top_k: int = typer.Option(10, help="top_k passed to the adapter config"),
    out: Path = typer.Option(Path(".ragci/run.json"), help="Where to write the run record"),
    concurrency: int = typer.Option(8, help="Concurrent cases"),
) -> None:
    """Measure retrieval quality against the golden set."""
    instance = load_adapter(adapter)
    metric_names = list(metric) if metric else [instance.__ragci_spec__.primary_metric]

    record = asyncio.run(
        run_cases(
            instance,
            list(load_golden(golden)),
            config={"top_k": top_k},
            metric_names=metric_names,
            golden_hash=golden_hash(golden),
            concurrency=concurrency,
        )
    )

    render_console(record, console=console)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_json(record, out)
    console.print(f"Run record written to {out}")

    if not record.valid:
        raise typer.Exit(code=2)


@app.command()
def gate(
    run: Path = typer.Option(Path(".ragci/run.json"), help="Run record to judge"),
    baseline: Path = typer.Option(Path(".ragci/baseline.json"), help="Baseline to compare against"),
    metric: str = typer.Option(None, help="Metric to gate on; defaults to the run's primary"),
    min_effect: float = typer.Option(
        DEFAULT_MIN_EFFECT, help="Smallest drop worth blocking a pull request over"
    ),
    alpha: float = typer.Option(DEFAULT_ALPHA, help="Significance level, one-sided"),
    markdown: Path = typer.Option(None, help="Write a markdown report here"),
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="Record this run as the new baseline and exit"
    ),
) -> None:
    """Fail the build when retrieval degrades, and only when the drop is real."""
    record = load_json(run)

    if update_baseline:
        Path(baseline).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run, baseline)
        console.print(f"Baseline updated from {run}.")
        return

    previous = load_json(baseline) if Path(baseline).exists() else None
    decision = decide(previous, record, metric=metric, min_effect=min_effect, alpha=alpha)

    style = "green" if decision.passed else "red"
    verdict = "Pass" if decision.passed else "Fail"
    console.print(f"[bold {style}]Gate: {verdict}[/] — {decision.message}")

    if markdown is not None:
        Path(markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown).write_text(render_markdown(record, decision), encoding="utf-8")
        console.print(f"Markdown report written to {markdown}")

    if decision.passed:
        return
    # Distinguish "the pipeline got worse" from "I could not tell": CI should treat a
    # broken comparison differently from a genuine quality regression.
    raise typer.Exit(code=1 if decision.reason == "regression" else 2)
