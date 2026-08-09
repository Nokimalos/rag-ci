"""A minimal, dependency-free adapter used to exercise rag-ci against itself.

Word-overlap ranking over four short documents. It is not a good retriever — it is a
predictable one, which is what a self-check needs.
"""

from ragci.contract import Chunk, ParamSpec, RetrievalTrace, Step, adapter

DOCS: dict[str, str] = {
    "solar": (
        "The Sun contains 99.8 percent of the mass of the solar system. "
        "Jupiter holds most of what remains. "
        "Mercury is the smallest planet and the closest to the Sun."
    ),
    "ocean": (
        "The Mariana Trench reaches nearly 11000 metres below sea level. "
        "Pressure there exceeds one thousand atmospheres. "
        "Only a handful of crewed descents have ever been made."
    ),
    "cities": (
        "Tokyo is the most populous metropolitan area on Earth. "
        "Delhi and Shanghai follow closely behind. "
        "Urban growth has slowed in most of Europe."
    ),
    "music": (
        "The piano was invented by Bartolomeo Cristofori around 1700. "
        "It replaced the harpsichord because it could vary loudness. "
        "The name comes from gravicembalo col piano e forte."
    ),
}


def _spans(text: str, chunk_size: int) -> list[tuple[int, int]]:
    return [(i, min(i + chunk_size, len(text))) for i in range(0, len(text), chunk_size)]


@adapter(
    index_time_params=[ParamSpec(name="chunk_size", values=[70, 140])],
    query_time_params=[ParamSpec(name="top_k", values=[3, 5])],
    primary_metric="recall@3",
)
class ReferenceRag:
    def retrieve(self, query: str, index, config: dict) -> RetrievalTrace:
        chunk_size = config.get("chunk_size", 70)
        top_k = config.get("top_k", 3)
        terms = set(query.lower().split())

        scored: list[tuple[float, Chunk]] = []
        for doc_id, text in DOCS.items():
            for start, end in _spans(text, chunk_size):
                body = text[start:end]
                score = len(terms & set(body.lower().split())) / len(terms) if terms else 0.0
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
        return RetrievalTrace(
            steps=[Step(query=query, chunks=[chunk for _, chunk in scored[:top_k]])],
            latency_ms=1.0,
        )
