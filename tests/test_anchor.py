"""Anchoring an existing Q/A set. The rule under test: never guess an offset."""

import json

import pytest

from ragci.anchor import QAPair, anchor, find_occurrences, load_pairs, suggest_passages
from ragci.corpus import Document


def _doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, text=text)


PHYSICS = _doc(
    "physics.txt",
    "Some preamble here.  Gravity is the attraction\nbetween masses. And a closing line.",
)


def test_an_answer_is_located_at_the_offsets_it_really_occupies():
    [passage] = find_occurrences(PHYSICS, "Gravity is the attraction between masses.")
    assert PHYSICS.text[passage.char_start : passage.char_end] == passage.text
    assert "Gravity is the attraction" in passage.text


def test_matching_survives_reflowed_whitespace_and_casing():
    # Copy-pasted answers pick up different line breaks and capitalisation. The offsets
    # must still address the document exactly as stored, newline included.
    [passage] = find_occurrences(PHYSICS, "gravity   IS the ATTRACTION between masses.")
    assert "\n" in PHYSICS.text[passage.char_start : passage.char_end]


def test_an_answer_that_is_not_there_is_not_located():
    assert find_occurrences(PHYSICS, "The moon is made of cheese.") == []


def test_a_single_match_becomes_an_anchored_case():
    pair = QAPair(id="q1", question="What is gravity?", answer="Gravity is the attraction")
    [outcome] = anchor([pair], [PHYSICS])

    assert outcome.status == "anchored"
    assert outcome.case is not None
    assert outcome.case.required_passages[0].doc_id == "physics.txt"
    assert outcome.case.reference_answer == pair.answer
    assert outcome.case.provenance == "anchored"


def test_an_answer_in_two_documents_is_ambiguous_not_arbitrarily_assigned():
    # Picking the first match would look tidy and quietly encode the wrong document.
    duplicate = _doc("notes.txt", "Gravity is the attraction between masses, restated.")
    pair = QAPair(id="q1", question="?", answer="Gravity is the attraction")
    [outcome] = anchor([pair], [PHYSICS, duplicate])

    assert outcome.status == "ambiguous"
    assert outcome.case is None
    assert {c.doc_id for c in outcome.candidates} == {"physics.txt", "notes.txt"}
    assert "2 times" in outcome.note


def test_two_occurrences_inside_one_document_are_also_ambiguous():
    twice = _doc("twice.txt", "The vault holds 400 charters. Later: the vault holds 400 charters.")
    pair = QAPair(id="q1", question="?", answer="the vault holds 400 charters")
    [outcome] = anchor([pair], [twice])
    assert outcome.status == "ambiguous"


def test_a_paraphrased_answer_is_suggested_rather_than_anchored():
    pair = QAPair(
        id="q1",
        question="What is gravity?",
        answer="masses attract each other — this attraction between masses is gravity",
    )
    [outcome] = anchor([pair], [PHYSICS])

    assert outcome.status == "not_found"
    assert outcome.case is None
    assert outcome.candidates, "a close sentence should be offered for review"
    assert "paraphrased" in outcome.note


def test_an_answer_from_a_different_corpus_says_so():
    pair = QAPair(id="q1", question="?", answer="The quarterly revenue was 4.2 million euros")
    [outcome] = anchor([pair], [PHYSICS])

    assert outcome.status == "not_found"
    assert outcome.candidates == []
    assert "right corpus" in outcome.note


def test_suggestions_are_whole_sentences_at_real_offsets():
    for passage in suggest_passages(PHYSICS, "the attraction between masses"):
        assert PHYSICS.text[passage.char_start : passage.char_end] == passage.text


def test_the_corpus_is_walked_once_however_many_pairs_there_are():
    # A pair-by-pair scan would re-read the corpus per question, which on a real corpus
    # is the difference between one pass and a thousand.
    walked = {"documents": 0}

    def corpus():
        for doc in (PHYSICS, _doc("other.txt", "Unrelated text.")):
            walked["documents"] += 1
            yield doc

    pairs = [
        QAPair(id=f"q{i}", question="?", answer="Gravity is the attraction") for i in range(20)
    ]
    anchor(pairs, corpus())
    assert walked["documents"] == 2


def test_pairs_load_with_generated_ids_when_none_are_given(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text(
        '{"question": "a?", "answer": "A"}\n\n{"id": "custom", "question": "b?", "answer": "B"}\n'
    )
    pairs = load_pairs(path)
    assert [p.id for p in pairs] == ["q0001", "custom"]


@pytest.mark.parametrize(
    "record", [{"question": "a?"}, {"answer": "A"}, {"question": "", "answer": "A"}]
)
def test_a_pair_missing_its_question_or_answer_is_rejected_with_the_line_number(tmp_path, record):
    path = tmp_path / "qa.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match=r":1 has no"):
        load_pairs(path)
