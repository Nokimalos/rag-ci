"""A deliberately trivial adapter: word-overlap ranking over three toy documents.

Its job is to make the whole pipeline testable offline, not to retrieve well.
"""

from ragci.contract import Chunk, ParamSpec, RetrievalTrace, Step, adapter

TOY_DOCS: dict[str, str] = {
    "physics": (
        "Gravity is the attraction between masses. "
        "Newton described it as a force acting at a distance. "
        "Einstein recast gravity as the curvature of spacetime."
    ),
    "biology": (
        "Photosynthesis converts light into chemical energy. "
        "Chloroplasts host the reaction inside plant cells. "
        "The process releases oxygen as a by-product."
    ),
    "history": (
        "The printing press spread literacy across Europe. "
        "Gutenberg assembled movable type around 1440. "
        "Book production costs collapsed within a generation."
    ),
}


def _split(text: str, chunk_size: int) -> list[tuple[int, int]]:
    return [(i, min(i + chunk_size, len(text))) for i in range(0, len(text), chunk_size)]


def _overlap(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().split())
    return len(q & t) / len(q) if q else 0.0


@adapter(
    index_time_params=[ParamSpec(name="chunk_size", values=[60, 120])],
    query_time_params=[ParamSpec(name="top_k", values=[1, 3, 5])],
    primary_metric="recall@3",
)
class ReferenceRag:
    def retrieve(self, query: str, index, config: dict) -> RetrievalTrace:
        chunk_size = config.get("chunk_size", 60)
        top_k = config.get("top_k", 5)

        scored: list[tuple[float, Chunk]] = []
        for doc_id, text in TOY_DOCS.items():
            for start, end in _split(text, chunk_size):
                body = text[start:end]
                score = _overlap(query, body)
                scored.append(
                    (
                        score,
                        Chunk(
                            text=body,
                            doc_id=doc_id,
                            char_start=start,
                            char_end=end,
                            score=score,
                        ),
                    )
                )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        chunks = [chunk for _, chunk in scored[:top_k]]
        return RetrievalTrace(steps=[Step(query=query, chunks=chunks)], latency_ms=1.0)
