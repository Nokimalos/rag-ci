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
