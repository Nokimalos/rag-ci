import pytest

from ragci.generate import GeneratedCase, generate_candidates
from ragci.golden import Passage


class FakeGenerator:
    """Deterministic stand-in — the suite must never need a network or an API key."""

    def __init__(self, confidence: float = 0.9, fail_on: set[str] | None = None):
        self.confidence = confidence
        self.fail_on = fail_on or set()
        self.seen: list[Passage] = []

    def generate(self, passage: Passage) -> GeneratedCase | None:
        self.seen.append(passage)
        if passage.doc_id in self.fail_on:
            return None
        return GeneratedCase(
            question=f"What does {passage.doc_id} say?",
            reference_answer=passage.text[:40],
            confidence=self.confidence,
        )


def _passage(doc_id: str = "d1", start: int = 0) -> Passage:
    return Passage(doc_id=doc_id, char_start=start, char_end=start + 50, text="X" * 50)


def test_each_passage_becomes_a_golden_case():
    cases = list(generate_candidates([_passage(), _passage("d2")], FakeGenerator()))
    assert len(cases) == 2
    assert cases[0].required_passages[0].doc_id == "d1"


def test_the_generated_case_is_anchored_to_the_source_passage():
    passage = _passage(start=120)
    case = next(iter(generate_candidates([passage], FakeGenerator())))
    required = case.required_passages[0]
    assert (required.char_start, required.char_end) == (120, 170)


def test_provenance_marks_the_case_as_unreviewed():
    case = next(iter(generate_candidates([_passage()], FakeGenerator())))
    assert case.provenance == "synthetic"


def test_confidence_is_carried_in_strata_for_review_ordering():
    case = next(iter(generate_candidates([_passage()], FakeGenerator(confidence=0.4))))
    assert case.strata["confidence"] == pytest.approx(0.4)


def test_case_ids_are_unique_and_stable():
    cases = list(generate_candidates([_passage("d1"), _passage("d2")], FakeGenerator()))
    assert len({c.id for c in cases}) == 2
    again = list(generate_candidates([_passage("d1"), _passage("d2")], FakeGenerator()))
    assert [c.id for c in cases] == [c.id for c in again]


def test_a_generator_returning_none_skips_that_passage():
    generator = FakeGenerator(fail_on={"d2"})
    cases = list(generate_candidates([_passage("d1"), _passage("d2")], generator))
    assert [c.required_passages[0].doc_id for c in cases] == ["d1"]
    assert len(generator.seen) == 2  # it was attempted, not silently skipped


def test_generation_is_lazy():
    generator = FakeGenerator()
    stream = generate_candidates([_passage("d1"), _passage("d2")], generator)
    next(stream)
    assert len(generator.seen) == 1


def test_confidence_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError):
        GeneratedCase(question="q", reference_answer="a", confidence=1.5)


def test_an_empty_question_is_rejected():
    with pytest.raises(ValueError):
        GeneratedCase(question="  ", reference_answer="a", confidence=0.5)


def test_importing_the_anthropic_generator_without_the_extra_is_actionable():
    import ragci.generate as gen

    if gen._anthropic_available():  # installed in this env — nothing to assert
        return
    with pytest.raises(RuntimeError, match=r"rag-ci\[generate\]"):
        gen.AnthropicGenerator()


class _FakeMessages:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)

        class Response:
            parsed_output = self.result

        return Response()


class _FakeClient:
    def __init__(self, result):
        self.messages = _FakeMessages(result)


def test_the_anthropic_generator_calls_parse_with_the_expected_shape():
    # Locks the SDK call shape: a typo in a keyword would otherwise only surface
    # against the real API, with a real bill attached.
    expected = GeneratedCase(question="How deep?", reference_answer="11000 m", confidence=0.8)
    client = _FakeClient(expected)

    from ragci.generate import DEFAULT_MODEL, AnthropicGenerator

    result = AnthropicGenerator(client=client).generate(_passage("ocean"))

    assert result == expected
    call = client.messages.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["output_format"] is GeneratedCase
    assert call["messages"][0]["role"] == "user"
    assert "ocean" in call["messages"][0]["content"]
    # Opus 5 rejects temperature/top_p, and prefill is unsupported — neither may appear.
    assert "temperature" not in call and "top_p" not in call
    assert all(m["role"] != "assistant" for m in call["messages"])


def test_the_model_is_configurable():
    client = _FakeClient(GeneratedCase(question="q", reference_answer="a", confidence=0.5))

    from ragci.generate import AnthropicGenerator

    AnthropicGenerator(model="claude-sonnet-5", client=client).generate(_passage())
    assert client.messages.calls[0]["model"] == "claude-sonnet-5"
