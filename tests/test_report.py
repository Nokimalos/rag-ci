import json

from rich.console import Console

from ragci.baseline import GateDecision
from ragci.report import load_json, render_console, render_markdown, save_json
from ragci.runner import CaseResult, RunRecord, Timings
from ragci.stats import MetricSummary


def _record(**overrides) -> RunRecord:
    defaults = dict(
        golden_hash="h",
        config={"top_k": 5},
        primary_metric="recall@5",
        metrics={"recall@5": MetricSummary(mean=0.75, ci_low=0.6, ci_high=0.9, n=40)},
        case_results=[CaseResult(case_id="q1", status="ok", scores={"recall@5": 1.0})],
        timings=Timings(latency_ms_p50=12.0, latency_ms_p95=30.0),
    )
    return RunRecord(**{**defaults, **overrides})


def test_console_shows_the_metric_and_its_interval():
    console = Console(record=True, width=100)
    render_console(_record(), console=console)
    output = console.export_text()
    assert "recall@5" in output
    assert "0.750" in output
    assert "0.600" in output and "0.900" in output


def test_console_warns_when_the_run_is_invalid():
    console = Console(record=True, width=100)
    render_console(_record(valid=False, error_rate=0.4), console=console)
    assert "INVALID" in console.export_text()


def test_console_warns_about_degraded_matching():
    console = Console(record=True, width=100)
    render_console(_record(degraded_matching=True), console=console)
    assert "degraded matching" in console.export_text().lower()


def test_console_survives_a_run_where_every_case_errored():
    # No case scored, so there is no metric summary to read a case count from.
    console = Console(record=True, width=100)
    render_console(_record(metrics={}, valid=False, error_rate=1.0), console=console)
    assert "INVALID" in console.export_text()


def test_console_shows_cost_and_tokens_when_present():
    console = Console(record=True, width=100)
    render_console(_record(cost_usd_per_query=0.0021, tokens_per_query=400.0), console=console)
    output = console.export_text()
    assert "0.0021" in output
    assert "400" in output


def test_json_round_trips(tmp_path):
    path = tmp_path / "run.json"
    save_json(_record(), path)
    assert load_json(path).metrics["recall@5"].mean == 0.75


def test_json_excludes_timings_so_identical_runs_diff_to_nothing(tmp_path):
    fast, slow = tmp_path / "fast.json", tmp_path / "slow.json"
    save_json(_record(timings=Timings(latency_ms_p50=1.0, latency_ms_p95=2.0)), fast)
    save_json(_record(timings=Timings(latency_ms_p50=900.0, latency_ms_p95=999.0)), slow)
    assert fast.read_text() == slow.read_text()


def test_a_reloaded_record_reports_no_latency_rather_than_zero(tmp_path):
    # save_json drops timings, so the gate — which always reads a record off disk — used
    # to render "latency p50 0 ms" into the pull request comment. A run measured at 900ms
    # was reported as instant.
    path = tmp_path / "run.json"
    save_json(_record(timings=Timings(latency_ms_p50=900.0, latency_ms_p95=999.0)), path)

    reloaded = load_json(path)
    assert reloaded.timings.latency_ms_p50 is None
    assert "latency" not in render_markdown(reloaded)


def test_markdown_reports_latency_when_the_record_is_still_in_memory():
    record = _record(timings=Timings(latency_ms_p50=900.0, latency_ms_p95=999.0))
    assert "latency p50 900 ms" in render_markdown(record)


def test_saved_json_is_indented_for_reviewable_diffs(tmp_path):
    path = tmp_path / "run.json"
    save_json(_record(), path)
    assert "\n  " in path.read_text()
    json.loads(path.read_text())


def _decision(**overrides) -> GateDecision:
    defaults = dict(
        passed=True,
        reason="ok",
        metric="recall@5",
        delta=0.004,
        p_value=0.42,
        ci_low=-0.01,
        ci_high=0.02,
        n_pairs=40,
        message="recall@5 moved by +0.004",
    )
    return GateDecision(**{**defaults, **overrides})


def test_markdown_contains_the_metric_table():
    output = render_markdown(_record())
    assert "| recall@5" in output
    assert "0.750" in output
    assert "[0.600, 0.900]" in output


def test_markdown_marks_a_pass():
    output = render_markdown(_record(), _decision())
    assert "✅" in output
    assert "40 paired cases" in output


def test_markdown_marks_a_regression():
    output = render_markdown(
        _record(),
        _decision(passed=False, reason="regression", delta=-0.08, p_value=0.001),
    )
    assert "❌" in output
    assert "-0.080" in output
    assert "0.0010" in output


def test_markdown_explains_a_stale_baseline_instead_of_showing_numbers():
    output = render_markdown(
        _record(),
        _decision(
            passed=False,
            reason="stale_baseline",
            delta=None,
            p_value=None,
            ci_low=None,
            ci_high=None,
            n_pairs=0,
            message="The golden set changed since the baseline was recorded.",
        ),
    )
    assert "golden set changed" in output
    assert "p=" not in output


def test_markdown_without_a_decision_omits_the_verdict_section():
    output = render_markdown(_record())
    assert "Gate" not in output


def test_markdown_flags_an_invalid_run():
    output = render_markdown(_record(valid=False, error_rate=0.5))
    assert "50%" in output
    assert "invalid" in output.lower()
