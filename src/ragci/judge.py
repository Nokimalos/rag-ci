"""Tier 2: is the generated answer actually grounded in what was retrieved?"""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, model_validator

from ragci.contract import Answer, Chunk

DEFAULT_MODEL = "claude-opus-5"


class Claim(BaseModel):
    text: str
    supported: bool
    supporting_chunk: int | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "Claim":
        if self.supported and self.supporting_chunk is None:
            raise ValueError("a supported claim must name its supporting_chunk")
        if not self.supported and self.supporting_chunk is not None:
            raise ValueError("an unsupported claim must not name a supporting_chunk")
        return self


class Verdict(BaseModel):
    claims: list[Claim]

    @property
    def faithfulness(self) -> float | None:
        """Fraction of the answer's claims the retrieved chunks actually support.

        Per-claim rather than a single boolean: knowing *which* sentence was invented
        is what makes the score actionable.

        None when the answer asserts nothing — an abstention is neither faithful nor
        unfaithful, and forcing it to a number distorts the mean either way.
        """
        if not self.claims:
            return None
        return sum(c.supported for c in self.claims) / len(self.claims)

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]


class Judge(Protocol):
    def assess(self, question: str, answer: Answer, chunks: list[Chunk]) -> Verdict | None: ...


def citation_accuracy(answer: Answer, verdict: Verdict, chunks: list[Chunk]) -> float | None:
    """Fraction of the answer's citations that actually support a claim.

    Two different lists are in play: `supporting_chunk` indexes the *retrieved* chunks
    the judge was shown, while citations point at whatever the answer chose to cite.
    Matching them by chunk identity keeps the score meaningful when an answer cites a
    subset of what was retrieved — which is the normal case.

    None, not zero, when the answer cites nothing: there is no citation accuracy to
    report, and zero would drag the mean down for a pipeline that simply does not cite.
    """
    if not answer.cited_chunks:
        return None

    used = {c.supporting_chunk for c in verdict.claims if c.supporting_chunk is not None}
    if any(index >= len(chunks) for index in used):
        raise ValueError("supporting_chunk index out of range for the retrieved chunks")

    supporting = {chunks[index].identity() for index in used}
    useful = sum(chunk.identity() in supporting for chunk in answer.cited_chunks)
    return useful / len(answer.cited_chunks)


SYSTEM_PROMPT = """\
You check whether an answer is grounded in the passages a retrieval system returned.

Break the answer into its individual factual claims. For each claim, decide whether the \
numbered passages support it, and if so which one. A claim is supported only when a \
passage states it or directly entails it — not when a passage is merely on the same topic, \
and not when the claim is true in general but absent from these passages.

Ignore hedging, restatements of the question, and offers to help further: those are not \
claims. Judge only what the answer asserts as fact. An answer that asserts nothing has no \
claims at all, which is a valid verdict."""


def _anthropic_available() -> bool:
    from ragci.generate import _anthropic_available as available

    return available()


def _render(chunks: list[Chunk]) -> str:
    # Numbered so the model can point at a passage by index rather than quoting it back.
    return "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks))


class AnthropicJudge:
    """Assesses grounding with a single call per case, at the model's default sampling.

    `temperature` is deliberately not passed: it is rejected with a 400 on claude-opus-5
    and the rest of the Claude 5 family, so there is no determinism knob to turn here.
    Judge variance is handled downstream instead — `calibrate` measures it by permuting
    passage order, and the paired bootstrap absorbs what remains.
    """

    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        if client is None and not _anthropic_available():
            raise RuntimeError(
                'Judging needs the anthropic SDK. Install it with: pip install "rag-ci[generate]"'
            )
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def assess(self, question: str, answer: Answer, chunks: list[Chunk]) -> Verdict | None:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Answer:\n{answer.text}\n\n"
                        f"Retrieved passages:\n{_render(chunks)}"
                    ),
                }
            ],
            output_format=Verdict,
        )
        return response.parsed_output


MAX_FLIP_RATE = 0.1


class CalibrationResult(BaseModel):
    cases: int
    flips: int
    flip_rate: float
    trustworthy: bool


def calibrate(
    judge: Judge,
    samples: Sequence[tuple[str, Answer, list[Chunk]]],
    *,
    max_flip_rate: float = MAX_FLIP_RATE,
) -> CalibrationResult:
    """Judge each sample twice, once with the chunk order reversed, and count flips.

    A judge whose verdict depends on the order of its inputs is not measuring grounding —
    it is measuring position. Better to learn that once, here, than to pay for it on
    every run and discover it when a gate fires for no reason.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("cannot calibrate: no samples supplied")

    flips = 0
    for question, answer, chunks in samples:
        first = judge.assess(question, answer, chunks)
        second = judge.assess(question, answer, list(reversed(chunks)))
        if first is None or second is None:
            flips += 1  # an unusable verdict is at least as bad as an unstable one
            continue
        if first.faithfulness != second.faithfulness:
            flips += 1

    rate = flips / len(samples)
    return CalibrationResult(
        cases=len(samples),
        flips=flips,
        flip_rate=rate,
        trustworthy=rate <= max_flip_rate,
    )
