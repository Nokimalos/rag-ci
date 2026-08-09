import json

import pytest

from ragci.corpus import CorpusError, Document, load_corpus


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
