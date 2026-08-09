"""Turning corpus passages into candidate golden cases.

The `anthropic` SDK is an optional extra: rag-ci's core measures retrieval and must
stay installable — and testable — without any model dependency.
"""

import hashlib
from collections.abc import Iterable, Iterator
from typing import Protocol

from pydantic import BaseModel, Field, field_validator

from ragci.golden import GoldenCase, Passage

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You write evaluation questions for a retrieval benchmark.

Given one passage from a document, write a single question whose answer is contained \
entirely within that passage. The question is used to test whether a search system can \
find this passage among thousands of others, so it must:

- be answerable from the passage alone, with no outside knowledge;
- be phrased the way a real user would ask it, not as a reference to the text (never \
"what does this passage say about X");
- contain enough distinctive detail to identify this passage rather than a generic one.

Also report your confidence that the question meets all three conditions. Use a low \
confidence when the passage is boilerplate, a fragment, or too generic to support a \
distinctive question — those cases are filtered out by a human reviewer, and a candid \
low score is more useful than an optimistic high one."""


class GeneratedCase(BaseModel):
    question: str
    reference_answer: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("question", "reference_answer")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class Generator(Protocol):
    def generate(self, passage: Passage) -> GeneratedCase | None: ...


def _anthropic_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


class AnthropicGenerator:
    """Generates one question per passage via the Anthropic API."""

    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        if client is None and not _anthropic_available():
            raise RuntimeError(
                "Question generation needs the anthropic SDK. "
                'Install it with: pip install "rag-ci[generate]"'
            )
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def generate(self, passage: Passage) -> GeneratedCase | None:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Document: {passage.doc_id}\n\nPassage:\n{passage.text}",
                }
            ],
            output_format=GeneratedCase,
        )
        return response.parsed_output


def _case_id(passage: Passage, prefix: str) -> str:
    digest = hashlib.sha256(
        f"{passage.doc_id}:{passage.char_start}:{passage.char_end}".encode()
    ).hexdigest()
    return f"{prefix}_{digest[:10]}"


def generate_candidates(
    passages: Iterable[Passage],
    generator: Generator,
    *,
    id_prefix: str = "q",
) -> Iterator[GoldenCase]:
    """Yield one unreviewed golden case per passage the generator could handle."""
    for passage in passages:
        generated = generator.generate(passage)
        if generated is None:
            continue
        yield GoldenCase(
            id=_case_id(passage, id_prefix),
            question=generated.question,
            required_passages=[passage],
            reference_answer=generated.reference_answer,
            provenance="synthetic",
            # Review order is driven by this; see `rag-ci golden review`.
            strata={"confidence": generated.confidence, "source": passage.doc_id},
        )
