"""Command line entry point."""

import asyncio
import importlib.util
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from ragci import __version__
from ragci.baseline import DEFAULT_ALPHA, DEFAULT_MIN_EFFECT, decide
from ragci.contract import AdapterSpec
from ragci.corpus import CorpusError, load_corpus, sample_with_report
from ragci.generate import DEFAULT_MODEL, AnthropicGenerator, generate_candidates
from ragci.golden import golden_hash, load_golden, save_golden
from ragci.passages import candidate_passages
from ragci.report import load_json, render_console, render_markdown, save_json
from ragci.review import ReviewSession, run_review
from ragci.runner import run_cases

app = typer.Typer(add_completion=False, help="Regression testing for RAG pipelines.")
console = Console()

TEMPLATE = Path(__file__).parent / "templates" / "adapter_template.py.txt"


def _version_callback(requested: bool) -> None:
    if requested:
        console.print(f"rag-ci {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Regression testing and configuration sweeps for RAG pipelines."""


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
        console.print(f"[red]Error:[/] no @adapter class found in {escape(str(path))}")
        raise typer.Exit(code=1)
    return candidates[0]()


@app.command()
def init(path: Path = typer.Option(Path("."), help="Where to write ragci_adapter.py")) -> None:
    """Scaffold an adapter file."""
    target = Path(path) / "ragci_adapter.py"
    if target.exists():
        console.print(f"[red]Error:[/] {escape(str(target))} already exists")
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
    console.print(f"[bold {style}]Gate: {verdict}[/] — {escape(decision.message)}")

    if markdown is not None:
        Path(markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown).write_text(render_markdown(record, decision), encoding="utf-8")
        console.print(f"Markdown report written to {markdown}")

    if decision.passed:
        return
    # Distinguish "the pipeline got worse" from "I could not tell": CI should treat a
    # broken comparison differently from a genuine quality regression.
    raise typer.Exit(code=1 if decision.reason == "regression" else 2)


golden_app = typer.Typer(help="Build and review the golden set.")
app.add_typer(golden_app, name="golden")


def _build_generator(model: str):
    """Seam for tests: the only place the CLI constructs a real generator."""
    return AnthropicGenerator(model=model)


@golden_app.command("gen")
def golden_gen(
    corpus: Path = typer.Option(..., help="Directory of .txt/.md files, or a .jsonl export"),
    out: Path = typer.Option(Path("golden.candidates.jsonl"), help="Where to write candidates"),
    sample: int = typer.Option(50, help="How many documents to sample"),
    stratify_by: list[str] = typer.Option(None, help="Metadata keys to stratify on"),
    model: str = typer.Option(DEFAULT_MODEL, help="Generation model"),
    seed: int = typer.Option(0, help="Sampling seed"),
    max_cases: int = typer.Option(None, help="Stop after this many candidates"),
) -> None:
    """Generate candidate questions from a corpus."""
    keys = tuple(stratify_by) if stratify_by else ("source",)
    try:
        documents, report = sample_with_report(load_corpus(corpus), n=sample, keys=keys, seed=seed)
    except CorpusError as exc:
        console.print(f"[red]Error:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Corpus: {report.total} documents across {report.strata} strata; sampled {report.sampled}."
    )

    try:
        generator = _build_generator(model)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    passages = [p for document in documents for p in candidate_passages(document)]
    console.print(f"{len(passages)} candidate passages. Generating with {model}...")

    cases = []
    for case in generate_candidates(passages, generator):
        cases.append(case)
        if max_cases is not None and len(cases) >= max_cases:
            break

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_golden(out, cases)
    console.print(
        f"Wrote {len(cases)} candidates to {out}. "
        f"Review them with: rag-ci golden review --candidates {out}"
    )


@golden_app.command("review")
def golden_review(
    candidates: Path = typer.Option(Path("golden.candidates.jsonl"), help="Candidate file"),
    golden: Path = typer.Option(Path("golden.jsonl"), help="Golden set to build"),
) -> None:
    """Accept, edit, or reject candidates - least confident first."""
    session = ReviewSession(candidates=candidates, golden=golden)
    if not session.pending():
        console.print("Nothing left to review.")
        return

    def prompt(case):
        answer = typer.prompt("[a]ccept / [e]dit / [r]eject / [s]kip / [q]uit", default="a")
        letter = answer.strip().lower()[:1]
        if letter == "e":
            return "accept", typer.prompt("Question", default=case.question)
        return {"a": "accept", "r": "reject", "s": "skip", "q": "quit"}.get(letter, "skip"), None

    stats = run_review(session, prompt=prompt, console=console)
    console.print(
        f"Reviewed {stats.reviewed}: {stats.accepted} accepted, {stats.rejected} rejected. "
        f"Golden set at {golden}."
    )
