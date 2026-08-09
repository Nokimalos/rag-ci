import pytest

from ragci.contract import Answer, Chunk
from ragci.judge import Claim, Verdict, citation_accuracy


def _chunk(doc_id: str = "d1") -> Chunk:
    return Chunk(text="Gravity curves spacetime.", doc_id=doc_id, char_start=0, char_end=25)


def test_faithfulness_is_the_supported_fraction():
    verdict = Verdict(
        claims=[
            Claim(text="a", supported=True, supporting_chunk=0),
            Claim(text="b", supported=True, supporting_chunk=0),
            Claim(text="c", supported=False, supporting_chunk=None),
        ]
    )
    assert verdict.faithfulness == pytest.approx(2 / 3)


def test_an_answer_with_no_claims_has_no_faithfulness_to_report():
    # "I don't know" makes no factual assertion. Scoring it 1.0 would reward saying
    # nothing; scoring it 0.0 would punish honest abstention as if it were a
    # hallucination. Neither is true — there is simply nothing to measure.
    assert Verdict(claims=[]).faithfulness is None


def test_unsupported_claims_are_listed_for_debugging():
    verdict = Verdict(
        claims=[
            Claim(text="real", supported=True, supporting_chunk=0),
            Claim(text="invented", supported=False, supporting_chunk=None),
        ]
    )
    assert [c.text for c in verdict.unsupported] == ["invented"]


def test_a_supported_claim_must_name_its_chunk():
    with pytest.raises(ValueError, match="supporting_chunk"):
        Claim(text="a", supported=True, supporting_chunk=None)


def test_an_unsupported_claim_must_not_name_a_chunk():
    with pytest.raises(ValueError, match="supporting_chunk"):
        Claim(text="a", supported=False, supporting_chunk=0)


def test_citation_accuracy_is_the_fraction_of_citations_that_support_something():
    answer = Answer(text="...", cited_chunks=[_chunk("d1"), _chunk("d2")])
    verdict = Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])
    # Chunk 0 of the retrieved list is d1; only that one of the two citations did work.
    retrieved = [_chunk("d1"), _chunk("d2")]
    assert citation_accuracy(answer, verdict, retrieved) == pytest.approx(0.5)


def test_citation_accuracy_is_none_when_nothing_is_cited():
    # Not zero: an answer that cites nothing has no citation accuracy to report.
    assert citation_accuracy(Answer(text="..."), Verdict(claims=[]), []) is None


def test_citation_accuracy_is_one_when_every_citation_is_used():
    answer = Answer(text="...", cited_chunks=[_chunk("d1")])
    verdict = Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])
    assert citation_accuracy(answer, verdict, [_chunk("d1")]) == 1.0


def test_citations_are_matched_against_the_retrieved_list_not_by_position():
    # The answer cites d2, which is chunk 1 of the retrieved list. The judge supported
    # chunk 0 (d1), so the citation did no work — a positional comparison would have
    # scored this 1.0 by accident.
    answer = Answer(text="...", cited_chunks=[_chunk("d2")])
    verdict = Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])
    assert citation_accuracy(answer, verdict, [_chunk("d1"), _chunk("d2")]) == 0.0


def test_a_supporting_chunk_out_of_range_is_rejected():
    answer = Answer(text="...", cited_chunks=[_chunk()])
    verdict = Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=7)])
    with pytest.raises(ValueError, match="out of range"):
        citation_accuracy(answer, verdict, [_chunk()])
