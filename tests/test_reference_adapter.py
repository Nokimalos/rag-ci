from tests.fixtures.reference_adapter import TOY_DOCS, ReferenceRag


def test_retrieve_returns_a_single_step_trajectory():
    trace = ReferenceRag().retrieve("gravity", index=None, config={"top_k": 3})
    assert len(trace.steps) == 1
    assert trace.steps[0].query == "gravity"


def test_retrieve_ranks_the_matching_document_first():
    trace = ReferenceRag().retrieve("photosynthesis", index=None, config={"top_k": 3})
    assert trace.all_chunks[0].doc_id == "biology"


def test_retrieve_honours_top_k():
    trace = ReferenceRag().retrieve("the", index=None, config={"top_k": 2})
    assert len(trace.all_chunks) == 2


def test_chunks_carry_offsets_that_index_back_into_the_source_document():
    trace = ReferenceRag().retrieve("gravity", index=None, config={"top_k": 1})
    chunk = trace.all_chunks[0]
    source = TOY_DOCS[chunk.doc_id]
    assert source[chunk.char_start : chunk.char_end] == chunk.text
