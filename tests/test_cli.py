import json
from pathlib import Path

from typer.testing import CliRunner

from ragci.cli import app
from ragci.golden import GoldenCase, Passage, save_golden

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
