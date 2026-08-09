"""Test doubles shared across modules."""

from ragci.generate import GeneratedCase
from ragci.golden import Passage


class FakeGenerator:
    def generate(self, passage: Passage) -> GeneratedCase | None:
        return GeneratedCase(
            question=f"What does {passage.doc_id} cover?",
            reference_answer=passage.text[:40],
            confidence=0.7,
        )


def install_fake_generator(monkeypatch) -> None:
    """Replace the CLI's generator factory so `golden gen` never touches the network."""
    import ragci.cli as cli

    monkeypatch.setattr(cli, "_build_generator", lambda model: FakeGenerator())


class FakeJudge:
    """Fixed verdict, or a failure — so tier-2 paths are testable with no model."""

    def __init__(self, fail: bool = False, faithfulness: float = 0.5):
        self.fail = fail
        self.faithfulness = faithfulness

    def assess(self, question, answer, chunks):
        from ragci.judge import Claim, Verdict

        if self.fail:
            return None
        supported = round(self.faithfulness * 2)
        claims = [Claim(text=f"c{i}", supported=True, supporting_chunk=0) for i in range(supported)]
        claims += [Claim(text=f"u{i}", supported=False) for i in range(2 - supported)]
        return Verdict(claims=claims)


class AnsweringRag:
    """ReferenceRag plus an answer(), so tier 2 has something to judge."""

    def __init__(self, cite: bool = True):
        from tests.fixtures.reference_adapter import ReferenceRag

        self._inner = ReferenceRag()
        self._cite = cite
        __class__.__ragci_spec__ = ReferenceRag.__ragci_spec__

    def retrieve(self, query, index, config):
        return self._inner.retrieve(query, index, config)

    def answer(self, query, trace, config):
        from ragci.contract import Answer

        chunks = trace.all_chunks
        return Answer(
            text="Gravity is the attraction between masses.",
            cited_chunks=chunks[:1] if self._cite else [],
            latency_ms=1.0,
        )
