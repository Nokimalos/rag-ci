"""Command line entry point."""

import asyncio
import importlib.util
import sys
from pathlib import Path

import typer
from rich.console import Console

from ragci.contract import AdapterSpec
from ragci.golden import golden_hash, load_golden
from ragci.report import render_console, save_json
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
