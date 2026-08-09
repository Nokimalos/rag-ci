import pytest

from ragci.contract import Answer, Chunk
from ragci.judge import Claim, Verdict, citation_accuracy


def _chunk(doc_id: str = "d1") -> Chunk:
    return Chunk(text="Gravity curves spacetime.", doc_id=doc_id, char_start=0, char_end=25)


def test_faithfulness_is_the_supported_fraction():
    verdict = Verdict(
        claims=[
            Claim(text="a", supported=True, supporting_chunk=0),
            Claim(text="b", supported=True, supporting_chunk=0),
            Claim(text="c", supported=False, supporting_chunk=None),
        ]
    )
    assert verdict.faithfulness == pytest.approx(2 / 3)


def test_an_answer_with_no_claims_has_no_faithfulness_to_report():
    # "I don't know" makes no factual assertion. Scoring it 1.0 would reward saying
    # nothing; scoring it 0.0 would punish honest abstention as if it were a
    # hallucination. Neither is true — there is simply nothing to measure.
    assert Verdict(claims=[]).faithfulness is None


def test_unsupported_claims_are_listed_for_debugging():
    verdict = Verdict(
        claims=[
            Claim(text="real", supported=True, supporting_chunk=0),
            Claim(text="invented", supported=False, supporting_chunk=None),
        ]
    )
    assert [c.text for c in verdict.unsupported] == ["invented"]


def test_a_supported_claim_must_name_its_chunk():
    with pytest.raises(ValueError, match="supporting_chunk"):
        Claim(text="a", supported=True, supporting_chunk=None)


def test_an_unsupported_claim_must_not_name_a_chunk():
    with pytest.raises(ValueError, match="supporting_chunk"):
        Claim(text="a", supported=False, supporting_chunk=0)


def test_citation_accuracy_is_the_fraction_of_citations_that_support_something():
    answer = Answer(text="...", cited_chunks=[_chunk("d1"), _chunk("d2")])
    verdict = Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])
    # Chunk 0 of the retrieved list is d1; only that one of the two citations did work.
    retrieved = [_chunk("d1"), _chunk("d2")]
    assert citation_accuracy(answer, verdict, retrieved) == pytest.approx(0.5)


def test_citation_accuracy_is_none_when_nothing_is_cited():
    # Not zero: an answer that cites nothing has no citation accuracy to report.
    assert citation_accuracy(Answer(text="..."), Verdict(claims=[]), []) is None


def test_citation_accuracy_is_one_when_every_citation_is_used():
    answer = Answer(text="...", cited_chunks=[_chunk("d1")])
    verdict = Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])
    assert citation_accuracy(answer, verdict, [_chunk("d1")]) == 1.0


def test_citations_are_matched_against_the_retrieved_list_not_by_position():
    # The answer cites d2, which is chunk 1 of the retrieved list. The judge supported
    # chunk 0 (d1), so the citation did no work — a positional comparison would have
    # scored this 1.0 by accident.
    answer = Answer(text="...", cited_chunks=[_chunk("d2")])
    verdict = Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])
    assert citation_accuracy(answer, verdict, [_chunk("d1"), _chunk("d2")]) == 0.0


def test_a_supporting_chunk_out_of_range_is_rejected():
    answer = Answer(text="...", cited_chunks=[_chunk()])
    verdict = Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=7)])
    with pytest.raises(ValueError, match="out of range"):
        citation_accuracy(answer, verdict, [_chunk()])


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


def test_the_judge_calls_parse_with_the_expected_shape():
    from ragci.judge import DEFAULT_MODEL, AnthropicJudge

    expected = Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])
    client = _FakeClient(expected)
    result = AnthropicJudge(client=client).assess(
        "How deep?", Answer(text="11000 m"), [_chunk("ocean")]
    )

    assert result == expected
    call = client.messages.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["output_format"] is Verdict
    # Opus 5 rejects temperature and prefill; neither may appear.
    assert "temperature" not in call and "top_p" not in call
    assert all(m["role"] != "assistant" for m in call["messages"])


def test_the_judge_sees_numbered_chunks_so_it_can_reference_them():
    from ragci.judge import AnthropicJudge

    client = _FakeClient(Verdict(claims=[]))
    AnthropicJudge(client=client).assess("q", Answer(text="a"), [_chunk("d1"), _chunk("d2")])
    content = client.messages.calls[0]["messages"][0]["content"]
    assert "[0]" in content and "[1]" in content


def test_the_judge_model_is_configurable():
    from ragci.judge import AnthropicJudge

    client = _FakeClient(Verdict(claims=[]))
    AnthropicJudge(model="claude-sonnet-5", client=client).assess("q", Answer(text="a"), [])
    assert client.messages.calls[0]["model"] == "claude-sonnet-5"


def test_using_the_judge_without_the_extra_is_actionable():
    import ragci.judge as judge_module

    if judge_module._anthropic_available():
        return
    with pytest.raises(RuntimeError, match=r"rag-ci\[generate\]"):
        judge_module.AnthropicJudge()


class _StableJudge:
    def assess(self, question, answer, chunks):
        return Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])


class _OrderSensitiveJudge:
    """Verdict depends on which chunk happens to come first — the failure to catch."""

    def assess(self, question, answer, chunks):
        if chunks and chunks[0].doc_id == "d1":
            return Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])
        return Verdict(claims=[Claim(text="a", supported=False)])


class _CountingJudge(_StableJudge):
    def __init__(self):
        self.calls = 0

    def assess(self, question, answer, chunks):
        self.calls += 1
        return super().assess(question, answer, chunks)


class _RecordingJudge(_StableJudge):
    def __init__(self):
        self.seen: list[list[str]] = []

    def assess(self, question, answer, chunks):
        self.seen.append([c.doc_id for c in chunks])
        return super().assess(question, answer, chunks)


class _HalfFlakyJudge:
    """Flips on half the samples, to exercise the threshold."""

    def __init__(self):
        self.sample = -1
        self.pass_index = 0

    def assess(self, question, answer, chunks):
        self.pass_index += 1
        if self.pass_index % 2 == 1:
            self.sample += 1
            return Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])
        if self.sample % 2 == 0:
            return Verdict(claims=[Claim(text="a", supported=False)])
        return Verdict(claims=[Claim(text="a", supported=True, supporting_chunk=0)])


def _samples(n: int):
    return [
        ("q", Answer(text="a", cited_chunks=[_chunk("d1")]), [_chunk("d1"), _chunk("d2")])
        for _ in range(n)
    ]


def test_a_stable_judge_never_flips():
    from ragci.judge import calibrate

    result = calibrate(_StableJudge(), _samples(10))
    assert result.flip_rate == 0.0
    assert result.trustworthy is True


def test_an_order_sensitive_judge_is_caught():
    from ragci.judge import calibrate

    # A judge whose verdict depends on input order is not measuring grounding.
    result = calibrate(_OrderSensitiveJudge(), _samples(10))
    assert result.flip_rate == 1.0
    assert result.trustworthy is False


def test_the_threshold_is_configurable():
    from ragci.judge import calibrate

    flaky = calibrate(_HalfFlakyJudge(), _samples(10), max_flip_rate=0.6)
    assert flaky.flip_rate == pytest.approx(0.5)
    assert flaky.trustworthy is True
    assert calibrate(_HalfFlakyJudge(), _samples(10), max_flip_rate=0.1).trustworthy is False


def test_each_sample_is_judged_twice():
    from ragci.judge import calibrate

    judge = _CountingJudge()
    calibrate(judge, _samples(4))
    assert judge.calls == 8


def test_the_permuted_pass_reverses_chunk_order():
    from ragci.judge import calibrate

    judge = _RecordingJudge()
    calibrate(judge, _samples(1))
    first, second = judge.seen
    assert second == list(reversed(first))


def test_calibrating_with_no_samples_is_rejected():
    from ragci.judge import calibrate

    with pytest.raises(ValueError, match="no samples"):
        calibrate(_StableJudge(), [])
