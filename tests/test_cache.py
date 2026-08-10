"""A cache that serves a stale verdict is worse than no cache. These are the guards."""

import pytest

from ragci import __version__
from ragci.cache import RunCache, digest_cases, fingerprint_file
from ragci.golden import GoldenCase, Passage
from ragci.runner import RunRecord
from ragci.stats import MetricSummary


def _cases(n: int = 3) -> list[GoldenCase]:
    return [
        GoldenCase(
            id=f"q{i}",
            question=f"question {i}",
            required_passages=[Passage(doc_id="d", char_start=0, char_end=4, text="text")],
        )
        for i in range(n)
    ]


def _record(mean: float = 0.75) -> RunRecord:
    return RunRecord(
        primary_metric="recall@5",
        metrics={"recall@5": MetricSummary(mean=mean, ci_low=mean, ci_high=mean, n=3)},
        case_results=[],
        golden_hash="h",
        config={},
    )


def _cache(tmp_path, *, fingerprint: str = "abc") -> RunCache:
    return RunCache(tmp_path / "cache", fingerprint=fingerprint, version=__version__)


def test_a_stored_record_comes_back(tmp_path):
    cache = _cache(tmp_path)
    key = cache.key(config={"top_k": 5}, cases=_cases(), metrics=["recall@5"])
    cache.put(key, _record())

    restored = cache.get(key)
    assert restored is not None
    assert restored.metrics["recall@5"].mean == 0.75
    assert cache.stats.hits == 1


def test_a_different_configuration_is_a_different_entry(tmp_path):
    cache = _cache(tmp_path)
    a = cache.key(config={"top_k": 5}, cases=_cases(), metrics=["recall@5"])
    b = cache.key(config={"top_k": 3}, cases=_cases(), metrics=["recall@5"])
    assert a != b


def test_editing_the_adapter_invalidates_everything(tmp_path):
    # The failure this prevents: change your retriever, re-run the sweep, and get the
    # old retriever's numbers back with no indication anything is wrong.
    before = _cache(tmp_path, fingerprint="v1")
    after = _cache(tmp_path, fingerprint="v2")
    arguments = dict(config={"top_k": 5}, cases=_cases(), metrics=["recall@5"])

    before.put(before.key(**arguments), _record())
    assert after.get(after.key(**arguments)) is None


def test_a_different_case_set_is_a_different_entry(tmp_path):
    cache = _cache(tmp_path)
    a = cache.key(config={}, cases=_cases(3), metrics=["recall@5"])
    b = cache.key(config={}, cases=_cases(4), metrics=["recall@5"])
    assert a != b


def test_reordered_cases_are_a_different_entry(tmp_path):
    # Order decides which cases a rung evaluates, so it has to be part of the key.
    cache = _cache(tmp_path)
    cases = _cases(3)
    a = cache.key(config={}, cases=cases, metrics=["recall@5"])
    b = cache.key(config={}, cases=list(reversed(cases)), metrics=["recall@5"])
    assert a != b


def test_a_different_rag_ci_version_is_a_different_entry(tmp_path):
    # A metric whose definition changed between releases must not be served from the
    # previous release's results.
    old = RunCache(tmp_path / "cache", fingerprint="abc", version="0.0.1")
    new = RunCache(tmp_path / "cache", fingerprint="abc", version=__version__)
    arguments = dict(config={}, cases=_cases(), metrics=["recall@5"])

    old.put(old.key(**arguments), _record())
    assert new.get(new.key(**arguments)) is None


def test_a_missing_entry_is_a_miss_not_an_error(tmp_path):
    cache = _cache(tmp_path)
    assert cache.get("nothing-here") is None
    assert cache.stats.misses == 1


def test_a_corrupt_entry_is_a_miss_rather_than_a_crash(tmp_path):
    cache = _cache(tmp_path)
    key = cache.key(config={}, cases=_cases(), metrics=["recall@5"])
    cache.put(key, _record())
    (cache.root / f"{key}.json").write_text("{ truncated")

    assert cache.get(key) is None  # re-measuring is always available
    assert cache.stats.misses == 1


def test_no_partial_file_is_left_behind(tmp_path):
    cache = _cache(tmp_path)
    cache.put(cache.key(config={}, cases=_cases(), metrics=["recall@5"]), _record())
    assert list(cache.root.glob("*.partial")) == []


def test_the_fingerprint_follows_the_file_contents(tmp_path):
    path = tmp_path / "adapter.py"
    path.write_text("x = 1")
    first = fingerprint_file(path)
    path.write_text("x = 2")
    assert fingerprint_file(path) != first


def test_the_case_digest_is_stable_across_calls():
    assert digest_cases(_cases()) == digest_cases(_cases())


@pytest.mark.parametrize("metrics", [["recall@5"], ["recall@5", "mrr"]])
def test_the_metric_set_is_part_of_the_key(tmp_path, metrics):
    cache = _cache(tmp_path)
    base = cache.key(config={}, cases=_cases(), metrics=["ndcg@10"])
    assert cache.key(config={}, cases=_cases(), metrics=metrics) != base
