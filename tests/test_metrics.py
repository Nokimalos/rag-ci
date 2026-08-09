import math

import pytest

from ragci.contract import Chunk, RetrievalTrace, Step
from ragci.golden import GoldenCase, Passage
from ragci.metrics import (
    all_passages_recall_at_k,
    mrr,
    ndcg_at_k,
    parse_metric_name,
    precision_at_k,
    recall_at_k,
)


def _passage(doc_id: str) -> Passage:
    return Passage(doc_id=doc_id, char_start=0, char_end=100, text=doc_id)


def _chunk(doc_id: str) -> Chunk:
    return Chunk(text=doc_id, doc_id=doc_id, char_start=0, char_end=100)


def _case(*doc_ids: str) -> GoldenCase:
    return GoldenCase(id="q", question="?", required_passages=[_passage(d) for d in doc_ids])


def _trace(*doc_ids: str) -> RetrievalTrace:
    return RetrievalTrace(steps=[Step(query="?", chunks=[_chunk(d) for d in doc_ids])])


def test_recall_counts_covered_passages():
    # Two required, one retrieved -> 0.5, computed by hand.
    assert recall_at_k(_trace("a", "z"), _case("a", "b"), k=2) == 0.5


def test_recall_respects_k():
    # "b" sits at rank 3, outside k=2.
    assert recall_at_k(_trace("z", "y", "b"), _case("b"), k=2) == 0.0


def test_all_passages_recall_is_all_or_nothing():
    assert all_passages_recall_at_k(_trace("a"), _case("a", "b"), k=5) == 0.0
    assert all_passages_recall_at_k(_trace("a", "b"), _case("a", "b"), k=5) == 1.0


def test_precision_counts_useful_chunks():
    # Three retrieved, one useful.
    assert precision_at_k(_trace("a", "y", "z"), _case("a"), k=3) == pytest.approx(1 / 3)


def test_precision_divides_by_retrieved_count_when_fewer_than_k():
    assert precision_at_k(_trace("a"), _case("a"), k=10) == 1.0


def test_mrr_uses_the_first_covering_rank():
    assert mrr(_trace("z", "a"), _case("a"), k=5) == 0.5
    assert mrr(_trace("a"), _case("a"), k=5) == 1.0


def test_mrr_is_zero_when_nothing_is_covered():
    assert mrr(_trace("y", "z"), _case("a"), k=5) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(_trace("a", "b"), _case("a", "b"), k=2) == pytest.approx(1.0)


def test_ndcg_penalises_a_relevant_result_placed_second():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k(_trace("z", "a"), _case("a"), k=2) == pytest.approx(expected)


def test_metrics_are_zero_on_an_empty_trace():
    empty = RetrievalTrace(steps=[])
    case = _case("a")
    assert recall_at_k(empty, case, k=5) == 0.0
    assert precision_at_k(empty, case, k=5) == 0.0
    assert mrr(empty, case, k=5) == 0.0
    assert ndcg_at_k(empty, case, k=5) == 0.0


def test_parse_metric_name():
    assert parse_metric_name("recall@10") == ("recall", 10)
    assert parse_metric_name("mrr") == ("mrr", 10)


def test_parse_metric_name_rejects_nonsense():
    with pytest.raises(ValueError, match="unknown metric"):
        parse_metric_name("banana@5")
