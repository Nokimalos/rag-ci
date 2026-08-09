import pytest
from pydantic import ValidationError

from ragci.golden import GoldenCase, Passage, golden_hash, load_golden, save_golden


def _case(case_id: str = "q1") -> GoldenCase:
    return GoldenCase(
        id=case_id,
        question="What curves spacetime?",
        required_passages=[
            Passage(doc_id="physics", char_start=0, char_end=40, text="Gravity is the attraction")
        ],
    )


def test_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "golden.jsonl"
    save_golden(path, [_case("q1"), _case("q2")])
    loaded = list(load_golden(path))
    assert [c.id for c in loaded] == ["q1", "q2"]
    assert loaded[0].required_passages[0].doc_id == "physics"


def test_a_case_must_have_at_least_one_required_passage():
    with pytest.raises(ValidationError):
        GoldenCase(id="q1", question="?", required_passages=[])


def test_load_is_lazy(tmp_path):
    # Reading the first case must not require parsing the rest of the file.
    path = tmp_path / "golden.jsonl"
    save_golden(path, [_case("q1")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ this line is not valid json\n")
    stream = load_golden(path)
    assert next(stream).id == "q1"


def test_hash_is_stable_across_writes(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    save_golden(a, [_case("q1"), _case("q2")])
    save_golden(b, [_case("q1"), _case("q2")])
    assert golden_hash(a) == golden_hash(b)


def test_hash_is_insensitive_to_case_order(tmp_path):
    # Reordering the file is not a semantic change; it must not invalidate a baseline.
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    save_golden(a, [_case("q1"), _case("q2")])
    save_golden(b, [_case("q2"), _case("q1")])
    assert golden_hash(a) == golden_hash(b)


def test_hash_changes_when_a_question_changes(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    save_golden(a, [_case("q1")])
    edited = _case("q1")
    edited.question = "Something else entirely?"
    save_golden(b, [edited])
    assert golden_hash(a) != golden_hash(b)
