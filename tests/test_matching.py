from ragci.contract import Chunk
from ragci.golden import Passage
from ragci.matching import covers, is_fallback_match

PASSAGE = Passage(doc_id="d1", char_start=100, char_end=200, text="alpha beta gamma delta")


def test_a_chunk_containing_the_passage_covers_it():
    assert covers(Chunk(text="x", doc_id="d1", char_start=50, char_end=250), PASSAGE)


def test_exact_overlap_covers():
    assert covers(Chunk(text="x", doc_id="d1", char_start=100, char_end=200), PASSAGE)


def test_overlap_at_the_threshold_covers():
    # 50 of 100 characters = exactly 0.5, which must pass.
    assert covers(Chunk(text="x", doc_id="d1", char_start=150, char_end=300), PASSAGE)


def test_overlap_below_the_threshold_does_not_cover():
    # 20 of 100 characters.
    assert not covers(Chunk(text="x", doc_id="d1", char_start=180, char_end=300), PASSAGE)


def test_identical_offsets_in_a_different_document_do_not_cover():
    assert not covers(Chunk(text="x", doc_id="d2", char_start=100, char_end=200), PASSAGE)


def test_adjacent_but_disjoint_chunk_does_not_cover():
    assert not covers(Chunk(text="x", doc_id="d1", char_start=200, char_end=300), PASSAGE)


def test_a_passage_straddling_two_chunks_is_covered_by_the_larger_half():
    first = Chunk(text="x", doc_id="d1", char_start=0, char_end=140)  # 40 of 100
    second = Chunk(text="x", doc_id="d1", char_start=140, char_end=280)  # 60 of 100
    assert not covers(first, PASSAGE)
    assert covers(second, PASSAGE)


def test_falls_back_to_token_overlap_when_the_chunk_has_no_offsets():
    chunk = Chunk(text="alpha beta gamma delta epsilon", doc_id="d1")
    assert covers(chunk, PASSAGE)
    assert is_fallback_match(chunk, PASSAGE)


def test_fallback_rejects_a_chunk_sharing_too_few_tokens():
    chunk = Chunk(text="alpha omega", doc_id="d1")
    assert not covers(chunk, PASSAGE)


def test_offset_matching_is_not_reported_as_fallback():
    chunk = Chunk(text="x", doc_id="d1", char_start=100, char_end=200)
    assert not is_fallback_match(chunk, PASSAGE)
