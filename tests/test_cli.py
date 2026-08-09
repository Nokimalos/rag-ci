import json
from pathlib import Path

from typer.testing import CliRunner

from ragci.cli import app
from ragci.golden import GoldenCase, Passage, save_golden
from ragci.report import save_json
from ragci.runner import CaseResult, RunRecord
from ragci.stats import MetricSummary

runner = CliRunner()

ADAPTER_SOURCE = """
from ragci.contract import Chunk, RetrievalTrace, Step, adapter

DOC = "Gravity is the attraction between masses and Einstein recast it as curvature."


@adapter(primary_metric="recall@3")
class TinyRag:
    def retrieve(self, query, index, config):
        return RetrievalTrace(
            steps=[
                Step(
                    query=query,
                    chunks=[Chunk(text=DOC, doc_id="physics", char_start=0, char_end=len(DOC))],
                )
            ],
            latency_ms=1.0,
        )
"""


def _write_workspace(tmp_path: Path) -> tuple[Path, Path]:
    adapter_path = tmp_path / "ragci_adapter.py"
    adapter_path.write_text(ADAPTER_SOURCE)
    golden_path = tmp_path / "golden.jsonl"
    save_golden(
        golden_path,
        [
            GoldenCase(
                id="q1",
                question="What curves spacetime?",
                required_passages=[
                    Passage(doc_id="physics", char_start=0, char_end=40, text="Gravity")
                ],
            )
        ],
    )
    return adapter_path, golden_path


def test_init_writes_an_adapter_template(tmp_path):
    result = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "ragci_adapter.py").exists()
    assert "def retrieve" in (tmp_path / "ragci_adapter.py").read_text()


def test_init_refuses_to_overwrite(tmp_path):
    (tmp_path / "ragci_adapter.py").write_text("# mine")
    result = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert result.exit_code != 0
    assert (tmp_path / "ragci_adapter.py").read_text() == "# mine"


def test_run_reports_metrics_and_writes_the_record(tmp_path):
    adapter_path, golden_path = _write_workspace(tmp_path)
    out = tmp_path / "run.json"
    result = runner.invoke(
        app,
        [
            "run",
            "--adapter",
            str(adapter_path),
            "--golden",
            str(golden_path),
            "--metric",
            "recall@3",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert "recall@3" in result.stdout
    payload = json.loads(out.read_text())
    assert payload["metrics"]["recall@3"]["mean"] == 1.0
    assert payload["golden_hash"]


def test_run_exits_nonzero_when_the_adapter_always_fails(tmp_path):
    _, golden_path = _write_workspace(tmp_path)
    broken = tmp_path / "broken_adapter.py"
    broken.write_text(
        "from ragci.contract import adapter\n"
        "@adapter()\n"
        "class Broken:\n"
        "    def retrieve(self, query, index, config):\n"
        "        raise RuntimeError('down')\n"
    )
    result = runner.invoke(app, ["run", "--adapter", str(broken), "--golden", str(golden_path)])
    assert result.exit_code == 2
    assert "INVALID" in result.stdout


def test_run_fails_clearly_when_the_module_has_no_adapter(tmp_path):
    _, golden_path = _write_workspace(tmp_path)
    plain = tmp_path / "plain.py"
    plain.write_text("x = 1\n")
    result = runner.invoke(app, ["run", "--adapter", str(plain), "--golden", str(golden_path)])
    assert result.exit_code != 0
    assert "no @adapter" in result.stdout


def test_run_uses_the_adapter_declared_primary_metric_by_default(tmp_path):
    adapter_path, golden_path = _write_workspace(tmp_path)
    result = runner.invoke(
        app, ["run", "--adapter", str(adapter_path), "--golden", str(golden_path)]
    )
    assert result.exit_code == 0
    assert "recall@3" in result.stdout


GATE_METRIC = "recall@10"


def _gate_record(offset: float, *, golden_hash: str = "h", valid: bool = True) -> RunRecord:
    scores = {f"q{i}": min(1.0, (i % 10) / 10 + offset) for i in range(120)}
    return RunRecord(
        golden_hash=golden_hash,
        config={},
        primary_metric=GATE_METRIC,
        metrics={
            GATE_METRIC: MetricSummary(
                mean=sum(scores.values()) / len(scores), ci_low=0.0, ci_high=1.0, n=len(scores)
            )
        },
        case_results=[
            CaseResult(case_id=cid, status="ok", scores={GATE_METRIC: s})
            for cid, s in scores.items()
        ],
        valid=valid,
    )


def test_gate_passes_on_an_identical_run(tmp_path):
    baseline, run = tmp_path / "baseline.json", tmp_path / "run.json"
    save_json(_gate_record(0.0), baseline)
    save_json(_gate_record(0.0), run)
    result = runner.invoke(app, ["gate", "--run", str(run), "--baseline", str(baseline)])
    assert result.exit_code == 0
    assert "Pass" in result.stdout


def test_gate_fails_on_a_regression(tmp_path):
    baseline, run = tmp_path / "baseline.json", tmp_path / "run.json"
    save_json(_gate_record(0.3), baseline)
    save_json(_gate_record(0.0), run)
    result = runner.invoke(app, ["gate", "--run", str(run), "--baseline", str(baseline)])
    assert result.exit_code == 1
    assert "Fail" in result.stdout


def test_gate_exits_two_when_the_baseline_is_stale(tmp_path):
    baseline, run = tmp_path / "baseline.json", tmp_path / "run.json"
    save_json(_gate_record(0.0, golden_hash="old"), baseline)
    save_json(_gate_record(0.0, golden_hash="new"), run)
    result = runner.invoke(app, ["gate", "--run", str(run), "--baseline", str(baseline)])
    assert result.exit_code == 2


def test_gate_passes_and_explains_when_there_is_no_baseline(tmp_path):
    run = tmp_path / "run.json"
    save_json(_gate_record(0.0), run)
    result = runner.invoke(
        app, ["gate", "--run", str(run), "--baseline", str(tmp_path / "absent.json")]
    )
    assert result.exit_code == 0
    assert "No baseline" in result.stdout


def test_gate_writes_the_markdown_report(tmp_path):
    baseline, run = tmp_path / "baseline.json", tmp_path / "run.json"
    md = tmp_path / "comment.md"
    save_json(_gate_record(0.0), baseline)
    save_json(_gate_record(0.0), run)
    runner.invoke(
        app,
        ["gate", "--run", str(run), "--baseline", str(baseline), "--markdown", str(md)],
    )
    assert "## rag-ci" in md.read_text()


def test_update_baseline_overwrites_and_does_not_judge(tmp_path):
    baseline, run = tmp_path / "baseline.json", tmp_path / "run.json"
    save_json(_gate_record(0.3), baseline)
    save_json(_gate_record(0.0), run)  # would be a regression
    result = runner.invoke(
        app,
        ["gate", "--run", str(run), "--baseline", str(baseline), "--update-baseline"],
    )
    assert result.exit_code == 0
    assert baseline.read_text() == run.read_text()


def test_min_effect_can_be_tightened_to_block_a_small_drop(tmp_path):
    baseline, run = tmp_path / "baseline.json", tmp_path / "run.json"
    base_record = _gate_record(0.3)
    tightened = _gate_record(0.3)
    for case in tightened.case_results:
        case.scores[GATE_METRIC] -= 0.005
    save_json(base_record, baseline)
    save_json(tightened, run)

    lenient = runner.invoke(app, ["gate", "--run", str(run), "--baseline", str(baseline)])
    strict = runner.invoke(
        app,
        ["gate", "--run", str(run), "--baseline", str(baseline), "--min-effect", "0.001"],
    )
    assert lenient.exit_code == 0
    assert strict.exit_code == 1
