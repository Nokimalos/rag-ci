import json

import pytest

from ragci.corpus import (
    CorpusError,
    Document,
    load_corpus,
    sample_with_report,
    stratified_sample,
)


def _write_tree(root):
    (root / "handbook").mkdir()
    (root / "handbook" / "leave.md").write_text("Annual leave is 25 days.", encoding="utf-8")
    (root / "handbook" / "pay.txt").write_text("Payday is the 25th.", encoding="utf-8")
    (root / "notes.md").write_text("Misc notes.", encoding="utf-8")
    (root / "ignore.pdf").write_bytes(b"%PDF-1.4 binary")


def test_loads_a_directory_of_text_files(tmp_path):
    _write_tree(tmp_path)
    docs = {d.doc_id: d for d in load_corpus(tmp_path)}
    assert set(docs) == {"handbook/leave.md", "handbook/pay.txt", "notes.md"}
    assert docs["notes.md"].text == "Misc notes."


def test_doc_ids_are_relative_posix_paths(tmp_path):
    # Stable across machines and OSes: the same corpus must hash the same everywhere.
    _write_tree(tmp_path)
    assert all("\\" not in d.doc_id for d in load_corpus(tmp_path))


def test_directory_metadata_records_the_parent_folder(tmp_path):
    _write_tree(tmp_path)
    docs = {d.doc_id: d for d in load_corpus(tmp_path)}
    assert docs["handbook/leave.md"].metadata["source"] == "handbook"
    assert docs["notes.md"].metadata["source"] == "."


def test_non_text_files_are_skipped(tmp_path):
    _write_tree(tmp_path)
    assert all(not d.doc_id.endswith(".pdf") for d in load_corpus(tmp_path))


def test_loads_a_jsonl_corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"doc_id": "a", "text": "Alpha.", "metadata": {"source": "wiki"}})
        + "\n"
        + json.dumps({"doc_id": "b", "text": "Beta."})
        + "\n",
        encoding="utf-8",
    )
    docs = list(load_corpus(path))
    assert [d.doc_id for d in docs] == ["a", "b"]
    assert docs[0].metadata["source"] == "wiki"
    assert docs[1].metadata == {}


def test_loading_is_lazy(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"doc_id": "a", "text": "Alpha."}) + "\n{ broken\n", encoding="utf-8"
    )
    assert next(load_corpus(path)).doc_id == "a"


def test_empty_documents_are_skipped(tmp_path):
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "real.txt").write_text("Content.", encoding="utf-8")
    assert [d.doc_id for d in load_corpus(tmp_path)] == ["real.txt"]


def test_a_missing_path_is_rejected_with_an_actionable_message(tmp_path):
    with pytest.raises(CorpusError, match="does not exist"):
        list(load_corpus(tmp_path / "absent"))


def test_a_directory_with_no_text_files_is_rejected(tmp_path):
    (tmp_path / "only.pdf").write_bytes(b"%PDF")
    with pytest.raises(CorpusError, match="no .txt or .md"):
        list(load_corpus(tmp_path))


def test_a_document_requires_a_non_empty_id():
    with pytest.raises(ValueError):
        Document(doc_id="", text="x")


def _docs(spec: dict[str, int]) -> list[Document]:
    return [
        Document(doc_id=f"{source}/{i}", text=f"text {i}", metadata={"source": source})
        for source, count in spec.items()
        for i in range(count)
    ]


def test_sample_is_capped_at_the_requested_size():
    assert len(stratified_sample(_docs({"a": 50, "b": 50}), n=10)) == 10


def test_sample_returns_everything_when_the_corpus_is_smaller_than_n():
    assert len(stratified_sample(_docs({"a": 3}), n=10)) == 3


def test_every_stratum_is_represented_even_when_tiny():
    # 999 documents in one source, 1 in another: uniform sampling would almost
    # certainly miss the small one entirely. Stratifying must not.
    sample = stratified_sample(_docs({"big": 999, "small": 1}), n=10)
    assert {d.metadata["source"] for d in sample} == {"big", "small"}


def test_remaining_slots_are_distributed_proportionally():
    sample = stratified_sample(_docs({"big": 90, "small": 10}), n=20)
    per_source: dict[str, int] = {}
    for doc in sample:
        per_source[doc.metadata["source"]] = per_source.get(doc.metadata["source"], 0) + 1
    assert per_source["big"] > per_source["small"]
    assert per_source["small"] >= 1


def test_a_stratum_smaller_than_its_allocation_does_not_waste_slots():
    # "small" can only supply 2; those spare slots must go to "big", not vanish.
    sample = stratified_sample(_docs({"big": 100, "small": 2}), n=20)
    assert len(sample) == 20


def test_sampling_is_reproducible():
    docs = _docs({"a": 40, "b": 40})
    assert [d.doc_id for d in stratified_sample(docs, n=12, seed=7)] == [
        d.doc_id for d in stratified_sample(docs, n=12, seed=7)
    ]


def test_different_seeds_select_different_documents():
    docs = _docs({"a": 40, "b": 40})
    assert [d.doc_id for d in stratified_sample(docs, n=12, seed=1)] != [
        d.doc_id for d in stratified_sample(docs, n=12, seed=2)
    ]


def test_multiple_stratum_keys_combine():
    docs = [
        Document(doc_id=f"{s}-{lang}-{i}", text="t", metadata={"source": s, "lang": lang})
        for s in ("a", "b")
        for lang in ("en", "fr")
        for i in range(10)
    ]
    sample = stratified_sample(docs, n=8, keys=("source", "lang"))
    seen = {(d.metadata["source"], d.metadata["lang"]) for d in sample}
    assert len(seen) == 4


def test_a_missing_metadata_key_falls_into_an_explicit_bucket():
    docs = [
        Document(doc_id="a", text="t"),
        Document(doc_id="b", text="t", metadata={"source": "x"}),
    ]
    assert len(stratified_sample(docs, n=2)) == 2


def test_more_strata_than_slots_still_respects_n():
    # 30 sources, 10 slots: one-per-stratum would return 30. n is a hard cap.
    docs = _docs({f"s{i}": 5 for i in range(30)})
    sample = stratified_sample(docs, n=10)
    assert len(sample) == 10
    assert len({d.metadata["source"] for d in sample}) == 10


def test_the_report_states_coverage():
    _, report = sample_with_report(_docs({"a": 60, "b": 40}), n=10)
    assert report.total == 100
    assert report.strata == 2
    assert report.sampled == 10
    assert sum(report.per_stratum.values()) == 10


def test_sampling_an_empty_corpus_is_rejected():
    with pytest.raises(CorpusError, match="no documents"):
        stratified_sample([], n=5)


def _big_corpus(n: int, sources: int = 4):
    """A generator, not a list: nothing here is ever all in memory at once."""
    for i in range(n):
        yield Document(
            doc_id=f"doc{i}",
            text="x" * 1000,
            metadata={"source": f"s{i % sources}"},
        )


def test_a_callable_source_is_read_twice_and_holds_no_text_between_passes():
    passes = []

    def read():
        passes.append(1)
        return _big_corpus(500)

    sample, report = sample_with_report(read, n=20, seed=0)
    assert len(passes) == 2  # count and stratify, then materialise only the winners
    assert report.total == 500
    assert len(sample) == 20


def test_the_callable_path_returns_the_same_sample_as_the_iterable_path():
    # The two-pass rewrite must not change which documents come back, or every golden
    # set generated before it would silently stop being reproducible.
    from_iterable, _ = sample_with_report(list(_big_corpus(200)), n=15, seed=7)
    from_callable, _ = sample_with_report(lambda: _big_corpus(200), n=15, seed=7)
    assert [d.doc_id for d in from_iterable] == [d.doc_id for d in from_callable]


def test_a_bare_generator_still_works_rather_than_looking_like_a_changed_corpus():
    # Walking an exhausted generator twice would find nothing on the second pass.
    sample, report = sample_with_report(_big_corpus(50), n=10, seed=0)
    assert len(sample) == 10
    assert report.total == 50


def test_a_corpus_that_shrinks_between_passes_is_reported_as_such():
    state = {"calls": 0}

    def shrinking():
        state["calls"] += 1
        n = 40 if state["calls"] == 1 else 5
        return _big_corpus(n)

    with pytest.raises(CorpusError, match="changed between the two passes"):
        sample_with_report(shrinking, n=20, seed=0)
