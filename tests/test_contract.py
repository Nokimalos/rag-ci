import pytest

from ragci.contract import Chunk, ParamSpec, RetrievalTrace, Step, adapter


def _chunk(doc_id: str, start: int, end: int) -> Chunk:
    return Chunk(text="x", doc_id=doc_id, char_start=start, char_end=end)


def test_all_chunks_flattens_steps_in_order():
    trace = RetrievalTrace(
        steps=[
            Step(query="a", chunks=[_chunk("d1", 0, 10)]),
            Step(query="b", chunks=[_chunk("d2", 0, 10)]),
        ]
    )
    assert [c.doc_id for c in trace.all_chunks] == ["d1", "d2"]


def test_all_chunks_deduplicates_across_steps_keeping_first_position():
    # A chunk retrieved by two sub-queries must not consume two top-k slots.
    repeated = _chunk("d1", 0, 10)
    trace = RetrievalTrace(
        steps=[
            Step(query="a", chunks=[repeated, _chunk("d2", 0, 10)]),
            Step(query="b", chunks=[repeated]),
        ]
    )
    assert [c.doc_id for c in trace.all_chunks] == ["d1", "d2"]


def test_chunks_without_offsets_deduplicate_on_text():
    a = Chunk(text="same", doc_id="d1")
    b = Chunk(text="same", doc_id="d1")
    trace = RetrievalTrace(steps=[Step(query="q", chunks=[a, b])])
    assert len(trace.all_chunks) == 1


def test_adapter_decorator_records_the_spec():
    @adapter(
        query_time_params=[ParamSpec(name="top_k", values=[5, 10])],
        primary_metric="recall@5",
    )
    class MyRag:
        def retrieve(self, query, index, config):
            return RetrievalTrace(steps=[])

    spec = MyRag.__ragci_spec__
    assert spec.primary_metric == "recall@5"
    assert spec.query_time_params[0].name == "top_k"
    assert spec.index_time_params == []


def test_adapter_decorator_rejects_a_class_without_retrieve():
    with pytest.raises(TypeError, match="must define retrieve"):

        @adapter()
        class Broken:
            pass
