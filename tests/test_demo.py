"""The demo is the first thing a stranger runs. It has to be true."""

import pytest
from rich.console import Console

from ragci.demo import BASELINE, CANDIDATE, METRIC, DemoRag, build_corpus, run_demo


def test_the_corpus_is_deterministic():
    first, first_cases = build_corpus()
    second, second_cases = build_corpus()
    assert [d.doc_id for d in first] == [d.doc_id for d in second]
    assert [c.question for c in first_cases] == [c.question for c in second_cases]


def test_there_are_enough_cases_for_the_statistics_to_speak():
    # Six questions cannot make any regression significant — that is the whole reason
    # the demo needed a corpus built for it rather than reusing examples/reference/.
    _, cases = build_corpus()
    assert len(cases) >= 100


def test_every_passage_is_anchored_where_it_says_it_is():
    documents, cases = build_corpus()
    by_id = {d.doc_id: d for d in documents}
    for case in cases:
        passage = case.required_passages[0]
        document = by_id[passage.doc_id]
        assert document.text[passage.char_start : passage.char_end] == passage.text


def test_the_questions_are_not_answerable_by_a_single_rare_word():
    # If each question matched exactly one document, recall would be 1.000 at every
    # setting and the demo would demonstrate nothing.
    documents, cases = build_corpus()
    question = cases[0].question
    terms = {w.lower().strip("?") for w in question.split() if len(w) > 3}
    competitors = sum(1 for d in documents if len(terms & set(d.text.lower().split())) >= 2)
    assert competitors > 1


async def test_the_demo_catches_the_regression_it_claims_to():
    # The guard that matters: if a change to the corpus makes this stop failing, the
    # demo would be showing a gate that does not catch anything.
    exit_code = await run_demo(Console(record=True, width=100))
    assert exit_code == 1


async def test_the_demo_regression_is_significant_and_not_just_large():
    from ragci.baseline import decide
    from ragci.runner import run_cases

    documents, cases = build_corpus()
    instance = DemoRag(documents)

    async def measure(config):
        return await run_cases(
            instance,
            cases,
            config=config,
            metric_names=[METRIC],
            golden_hash="demo",
            index=instance.build_index(None, config),
        )

    decision = decide(await measure(BASELINE), await measure(CANDIDATE), metric=METRIC)
    assert decision.p_value < 0.01
    assert decision.delta is not None and decision.delta < -0.05
    assert decision.n_pairs >= 100


async def test_the_demo_output_shows_the_evidence_not_just_the_verdict():
    console = Console(record=True, width=100)
    await run_demo(console)

    out = " ".join(console.export_text().split())
    assert "144 paired cases" in out
    assert "95% CI" in out
    assert "REGRESSION" in out


def test_the_demo_adapter_declares_a_grid_so_sweep_works_on_it():
    spec = DemoRag.__ragci_spec__
    assert spec.primary_metric == METRIC
    assert [p.name for p in spec.index_time_params] == ["chunk_size"]


@pytest.mark.parametrize("config", [BASELINE, CANDIDATE])
def test_both_configurations_are_inside_the_declared_grid(config):
    # A demo that gated on a value the adapter does not declare could not be reproduced
    # with `rag-ci sweep`.
    spec = DemoRag.__ragci_spec__
    declared = {p.name: p.values for p in spec.index_time_params + spec.query_time_params}
    assert all(value in declared[name] for name, value in config.items())
