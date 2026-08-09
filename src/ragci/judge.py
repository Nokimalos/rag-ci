"""Tier 2: is the generated answer actually grounded in what was retrieved?"""

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
