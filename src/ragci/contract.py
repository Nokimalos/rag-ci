"""The contract between rag-ci and a user's RAG pipeline."""

from typing import Any

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Chunk(BaseModel):
    text: str
    doc_id: str
    char_start: int | None = None
    char_end: int | None = None
    score: float | None = None

    def identity(self) -> tuple[str, int | None, int | None, str]:
        """Offsets identify a chunk when present; text is the fallback key."""
        if self.char_start is not None and self.char_end is not None:
            return (self.doc_id, self.char_start, self.char_end, "")
        return (self.doc_id, None, None, self.text)


class Step(BaseModel):
    query: str
    chunks: list[Chunk] = Field(default_factory=list)


class RetrievalTrace(BaseModel):
    steps: list[Step] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens: TokenUsage | None = None
    cost_usd: float | None = None

    @property
    def all_chunks(self) -> list[Chunk]:
        """Chunks across every step, deduplicated, first occurrence keeping its rank.

        Multi-hop pipelines routinely retrieve the same chunk from several sub-queries;
        counting it twice would let one chunk occupy several top-k slots.
        """
        seen: set[tuple[str, int | None, int | None, str]] = set()
        ordered: list[Chunk] = []
        for step in self.steps:
            for chunk in step.chunks:
                key = chunk.identity()
                if key not in seen:
                    seen.add(key)
                    ordered.append(chunk)
        return ordered


class Answer(BaseModel):
    text: str
    cited_chunks: list[Chunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens: TokenUsage | None = None
    cost_usd: float | None = None


class ParamSpec(BaseModel):
    name: str
    values: list[Any]


class AdapterSpec(BaseModel):
    index_time_params: list[ParamSpec] = Field(default_factory=list)
    query_time_params: list[ParamSpec] = Field(default_factory=list)
    primary_metric: str = "recall@10"


def adapter(
    *,
    index_time_params: list[ParamSpec] | None = None,
    query_time_params: list[ParamSpec] | None = None,
    primary_metric: str = "recall@10",
):
    """Mark a class as a rag-ci adapter and record its sweepable parameters."""

    def decorate(cls: type) -> type:
        if not callable(getattr(cls, "retrieve", None)):
            raise TypeError(f"{cls.__name__} must define retrieve(self, query, index, config)")
        cls.__ragci_spec__ = AdapterSpec(
            index_time_params=index_time_params or [],
            query_time_params=query_time_params or [],
            primary_metric=primary_metric,
        )
        return cls

    return decorate
