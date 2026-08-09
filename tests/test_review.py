from ragci.golden import GoldenCase, Passage, load_golden, save_golden
from ragci.review import ReviewSession, review_order, run_review


def _case(case_id: str, confidence: float) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        question=f"Question {case_id}?",
        required_passages=[Passage(doc_id="d", char_start=0, char_end=10, text="0123456789")],
        provenance="synthetic",
        strata={"confidence": confidence},
    )


def _workspace(tmp_path, cases):
    candidates, golden = tmp_path / "candidates.jsonl", tmp_path / "golden.jsonl"
    save_golden(candidates, cases)
    return ReviewSession(candidates=candidates, golden=golden)


def test_lowest_confidence_is_reviewed_first():
    ordered = review_order([_case("a", 0.9), _case("b", 0.2), _case("c", 0.5)])
    assert [c.id for c in ordered] == ["b", "c", "a"]


def test_cases_without_a_confidence_score_are_reviewed_first():
    plain = _case("x", 0.5)
    plain.strata = {}
    assert review_order([_case("a", 0.1), plain])[0].id == "x"


def test_accepting_writes_the_case_to_the_golden_set(tmp_path):
    session = _workspace(tmp_path, [_case("a", 0.5)])
    run_review(session, prompt=lambda case: ("accept", None), console=None)
    assert [c.id for c in load_golden(session.golden)] == ["a"]


def test_an_accepted_case_is_marked_reviewed(tmp_path):
    session = _workspace(tmp_path, [_case("a", 0.5)])
    run_review(session, prompt=lambda case: ("accept", None), console=None)
    accepted = next(iter(load_golden(session.golden)))
    assert accepted.provenance == "synthetic-reviewed"
    assert accepted.reviewed_at


def test_rejecting_writes_nothing(tmp_path):
    session = _workspace(tmp_path, [_case("a", 0.5)])
    run_review(session, prompt=lambda case: ("reject", None), console=None)
    assert not session.golden.exists() or list(load_golden(session.golden)) == []


def test_editing_replaces_the_question(tmp_path):
    session = _workspace(tmp_path, [_case("a", 0.5)])
    run_review(session, prompt=lambda case: ("accept", "A better question?"), console=None)
    assert next(iter(load_golden(session.golden))).question == "A better question?"


def test_quitting_stops_but_keeps_earlier_decisions(tmp_path):
    session = _workspace(tmp_path, [_case("a", 0.1), _case("b", 0.2), _case("c", 0.3)])
    seen: list[str] = []

    def prompt(case):
        seen.append(case.id)
        return ("quit", None) if case.id == "b" else ("accept", None)

    run_review(session, prompt=prompt, console=None)
    assert seen == ["a", "b"]
    assert [c.id for c in load_golden(session.golden)] == ["a"]


def test_review_resumes_where_it_stopped(tmp_path):
    session = _workspace(tmp_path, [_case("a", 0.1), _case("b", 0.2)])
    run_review(
        session,
        prompt=lambda case: ("quit", None) if case.id == "b" else ("accept", None),
        console=None,
    )
    resumed = ReviewSession(candidates=session.candidates, golden=session.golden)
    assert [c.id for c in resumed.pending()] == ["b"]


def test_skipping_leaves_the_case_pending(tmp_path):
    session = _workspace(tmp_path, [_case("a", 0.5)])
    run_review(session, prompt=lambda case: ("skip", None), console=None)
    resumed = ReviewSession(candidates=session.candidates, golden=session.golden)
    assert [c.id for c in resumed.pending()] == ["a"]


def test_a_rejected_case_does_not_come_back(tmp_path):
    session = _workspace(tmp_path, [_case("a", 0.1), _case("b", 0.2)])
    run_review(session, prompt=lambda case: ("reject", None), console=None)
    resumed = ReviewSession(candidates=session.candidates, golden=session.golden)
    assert resumed.pending() == []


def test_stats_report_the_session(tmp_path):
    session = _workspace(tmp_path, [_case("a", 0.1), _case("b", 0.2)])
    stats = run_review(
        session,
        prompt=lambda case: ("accept", None) if case.id == "a" else ("reject", None),
        console=None,
    )
    assert (stats.accepted, stats.rejected, stats.reviewed) == (1, 1, 2)
