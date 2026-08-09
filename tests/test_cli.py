import json
from pathlib import Path

from typer.testing import CliRunner

from ragci.cli import app
from ragci.golden import GoldenCase, Passage, save_golden
from ragci.report import save_json
from ragci.runner import CaseResult, RunRecord
from ragci.stats import MetricSummary

runner = CliRunner()


def _out(result) -> str:
    """stdout with wrapping collapsed: rich breaks lines at terminal width,
    which differs between a laptop and a CI runner with longer tmp paths."""
    return " ".join(result.stdout.split())


ADAPTER_SOURCE = """
from ragci.contract import Chunk, ParamSpec, RetrievalTrace, Step, adapter

DOC = "Gravity is the attraction between masses and Einstein recast it as curvature."


@adapter(
    query_time_params=[ParamSpec(name="top_k", values=[1, 3])],
    primary_metric="recall@3",
)
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
    assert "recall@3" in _out(result)
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
    assert "INVALID" in _out(result)


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
    assert "recall@3" in _out(result)


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
    assert "Pass" in _out(result)


def test_gate_fails_on_a_regression(tmp_path):
    baseline, run = tmp_path / "baseline.json", tmp_path / "run.json"
    save_json(_gate_record(0.3), baseline)
    save_json(_gate_record(0.0), run)
    result = runner.invoke(app, ["gate", "--run", str(run), "--baseline", str(baseline)])
    assert result.exit_code == 1
    assert "Fail" in _out(result)


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
    assert "No baseline" in _out(result)


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


def test_version_flag_prints_the_package_version():
    import ragci

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert ragci.__version__ in result.stdout


def test_version_flag_does_not_require_an_adapter(tmp_path, monkeypatch):
    # --version must work from any directory, with no adapter and no golden set in sight.
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["--version"]).exit_code == 0


def test_golden_gen_reports_coverage_and_writes_candidates(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "a").mkdir(parents=True)
    (corpus / "b").mkdir()
    for i in range(4):
        (corpus / "a" / f"{i}.md").write_text("Alpha " * 60, encoding="utf-8")
    (corpus / "b" / "only.md").write_text("Beta " * 60, encoding="utf-8")

    from tests.fakes import install_fake_generator

    install_fake_generator(monkeypatch)

    out = tmp_path / "candidates.jsonl"
    result = runner.invoke(
        app, ["golden", "gen", "--corpus", str(corpus), "--out", str(out), "--sample", "5"]
    )
    assert result.exit_code == 0
    assert "2 strata" in _out(result)
    assert out.exists()


def test_golden_gen_without_the_anthropic_extra_says_how_to_install(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("Alpha " * 60, encoding="utf-8")

    import ragci.generate as gen

    monkeypatch.setattr(gen, "_anthropic_available", lambda: False)
    result = runner.invoke(app, ["golden", "gen", "--corpus", str(corpus)])
    assert result.exit_code != 0
    assert "rag-ci[generate]" in _out(result)


def test_golden_gen_rejects_a_missing_corpus(tmp_path):
    result = runner.invoke(app, ["golden", "gen", "--corpus", str(tmp_path / "nope")])
    assert result.exit_code != 0
    assert "does not exist" in _out(result)


def test_golden_review_accepts_and_writes(tmp_path):
    from ragci.golden import GoldenCase, Passage, load_golden, save_golden

    candidates, golden = tmp_path / "candidates.jsonl", tmp_path / "golden.jsonl"
    save_golden(
        candidates,
        [
            GoldenCase(
                id="q1",
                question="What is alpha?",
                required_passages=[
                    Passage(doc_id="a.md", char_start=0, char_end=20, text="A" * 20)
                ],
                provenance="synthetic",
                strata={"confidence": 0.5},
            )
        ],
    )
    result = runner.invoke(
        app,
        ["golden", "review", "--candidates", str(candidates), "--golden", str(golden)],
        input="a\n",
    )
    assert result.exit_code == 0
    assert [c.id for c in load_golden(golden)] == ["q1"]


def test_golden_review_with_nothing_pending_says_so(tmp_path):
    from ragci.golden import GoldenCase, Passage, save_golden

    candidates, golden = tmp_path / "candidates.jsonl", tmp_path / "golden.jsonl"
    case = GoldenCase(
        id="q1",
        question="What is alpha?",
        required_passages=[Passage(doc_id="a.md", char_start=0, char_end=20, text="A" * 20)],
    )
    save_golden(candidates, [case])
    save_golden(golden, [case])
    result = runner.invoke(
        app, ["golden", "review", "--candidates", str(candidates), "--golden", str(golden)]
    )
    assert result.exit_code == 0
    assert "nothing" in _out(result).lower()


def test_sweep_reports_a_winner_and_the_saving(tmp_path):
    adapter_path, golden_path = _write_workspace(tmp_path)
    result = runner.invoke(
        app,
        [
            "sweep",
            "--adapter",
            str(adapter_path),
            "--golden",
            str(golden_path),
            "--metric",
            "recall@3",
            "--out",
            str(tmp_path / "sweep.json"),
        ],
    )
    assert result.exit_code == 0
    assert "winner" in _out(result).lower()
    assert (tmp_path / "sweep.json").exists()


def test_sweep_states_how_much_of_the_grid_it_skipped(tmp_path):
    adapter_path, golden_path = _write_workspace(tmp_path)
    result = runner.invoke(
        app,
        [
            "sweep",
            "--adapter",
            str(adapter_path),
            "--golden",
            str(golden_path),
            "--metric",
            "recall@3",
        ],
    )
    # Silent truncation reads as full coverage; the cost must be on screen.
    assert "case-evaluations" in _out(result)


def test_sweep_can_be_restricted_to_named_parameters(tmp_path):
    adapter_path, golden_path = _write_workspace(tmp_path)
    result = runner.invoke(
        app,
        [
            "sweep",
            "--adapter",
            str(adapter_path),
            "--golden",
            str(golden_path),
            "--metric",
            "recall@3",
            "--only",
            "top_k",
        ],
    )
    assert result.exit_code == 0


def test_sweep_rejects_an_undeclared_parameter(tmp_path):
    adapter_path, golden_path = _write_workspace(tmp_path)
    result = runner.invoke(
        app,
        [
            "sweep",
            "--adapter",
            str(adapter_path),
            "--golden",
            str(golden_path),
            "--only",
            "nonexistent",
        ],
    )
    assert result.exit_code != 0
    assert "not declared" in _out(result)


JUDGING_ADAPTER = """
from ragci.contract import Answer, Chunk, ParamSpec, RetrievalTrace, Step, adapter

DOC = "Gravity is the attraction between masses and Einstein recast it as curvature."


@adapter(query_time_params=[ParamSpec(name="top_k", values=[1, 3])], primary_metric="recall@3")
class AnsweringRag:
    def retrieve(self, query, index, config):
        chunk = Chunk(text=DOC, doc_id="physics", char_start=0, char_end=len(DOC))
        return RetrievalTrace(steps=[Step(query=query, chunks=[chunk])], latency_ms=1.0)

    def answer(self, query, trace, config):
        return Answer(text=DOC, cited_chunks=trace.all_chunks[:1], latency_ms=1.0)
"""


def _judging_workspace(tmp_path):
    adapter_path = tmp_path / "ragci_adapter.py"
    adapter_path.write_text(JUDGING_ADAPTER)
    _, golden_path = _write_workspace(tmp_path)
    adapter_path.write_text(JUDGING_ADAPTER)  # _write_workspace overwrote it
    return adapter_path, golden_path


def _install_fake_judge(monkeypatch):
    import ragci.cli as cli
    from tests.fakes import FakeJudge

    monkeypatch.setattr(cli, "_build_judge", lambda model: FakeJudge())


def test_run_with_judge_reports_faithfulness(tmp_path, monkeypatch):
    adapter_path, golden_path = _judging_workspace(tmp_path)
    _install_fake_judge(monkeypatch)
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
            "--judge",
            "--out",
            str(tmp_path / "run.json"),
        ],
    )
    assert result.exit_code == 0
    assert "faithfulness" in _out(result)


def test_run_without_judge_reports_no_tier_two(tmp_path):
    adapter_path, golden_path = _judging_workspace(tmp_path)
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
        ],
    )
    assert result.exit_code == 0
    assert "faithfulness" not in _out(result)


def test_judge_calibrate_passes_a_stable_judge(tmp_path, monkeypatch):
    adapter_path, golden_path = _judging_workspace(tmp_path)
    _install_fake_judge(monkeypatch)
    result = runner.invoke(
        app,
        ["judge", "calibrate", "--adapter", str(adapter_path), "--golden", str(golden_path)],
    )
    assert result.exit_code == 0
    assert "flip rate" in _out(result).lower()


def test_judge_calibrate_fails_an_unstable_judge(tmp_path, monkeypatch):
    adapter_path, golden_path = _judging_workspace(tmp_path)

    class Unstable:
        def __init__(self):
            self.n = 0

        def assess(self, question, answer, chunks):
            from ragci.judge import Claim, Verdict

            self.n += 1
            if self.n % 2 == 0:
                return Verdict(claims=[Claim(text="a", supported=False)])
            return Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])

    import ragci.cli as cli

    monkeypatch.setattr(cli, "_build_judge", lambda model: Unstable())
    result = runner.invoke(
        app,
        ["judge", "calibrate", "--adapter", str(adapter_path), "--golden", str(golden_path)],
    )
    assert result.exit_code != 0
    assert "not trustworthy" in _out(result).lower()
