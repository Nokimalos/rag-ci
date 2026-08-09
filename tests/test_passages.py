from ragci.corpus import Document
from ragci.passages import candidate_passages

BODY = (
    "Short.\n\n"
    + "A" * 200
    + "\n\n"
    + "The Mariana Trench reaches nearly 11000 metres below sea level, deeper than "
    "Everest is tall, and the pressure there exceeds one thousand atmospheres.\n"
)


def _doc(text: str = BODY) -> Document:
    return Document(doc_id="d1", text=text)


def test_offsets_index_back_into_the_source_exactly():
    document = _doc()
    for passage in candidate_passages(document):
        assert document.text[passage.char_start : passage.char_end] == passage.text


def test_short_paragraphs_are_dropped():
    assert all(len(p.text) >= 120 for p in candidate_passages(_doc()))


def test_every_passage_carries_the_document_id():
    assert all(p.doc_id == "d1" for p in candidate_passages(_doc()))


def test_a_document_with_no_blank_lines_still_yields_a_passage():
    document = _doc("B" * 400)
    passages = candidate_passages(document)
    assert len(passages) == 1
    assert passages[0].char_start == 0


def test_an_overlong_paragraph_is_split_at_a_sentence_boundary():
    sentence = "This sentence is exactly long enough to matter in the split. "
    document = _doc(sentence * 40)
    passages = candidate_passages(document, max_chars=500)
    assert all(len(p.text) <= 500 for p in passages)
    assert all(document.text[p.char_start : p.char_end] == p.text for p in passages)


def test_passages_do_not_overlap_and_keep_document_order():
    passages = candidate_passages(_doc())
    starts = [p.char_start for p in passages]
    assert starts == sorted(starts)
    for earlier, later in zip(passages, passages[1:], strict=False):
        assert earlier.char_end <= later.char_start


def test_a_document_shorter_than_the_minimum_yields_nothing():
    assert candidate_passages(_doc("Too short.")) == []


def test_leading_and_trailing_whitespace_is_excluded_from_the_span():
    document = _doc("\n\n   " + "C" * 200 + "   \n\n")
    passage = candidate_passages(document)[0]
    assert passage.text == passage.text.strip()
    assert document.text[passage.char_start : passage.char_end] == passage.text
