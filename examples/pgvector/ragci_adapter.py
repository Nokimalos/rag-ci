"""rag-ci against Postgres + pgvector, with no framework in between.

Deliberately plain psycopg: the point is to show where the offsets live — two integer
columns next to the vector — so this transfers whether you reach pgvector through
SQLAlchemy, LangChain, or raw SQL like here.

Set DATABASE_URL, or run the container in the README.
"""

import os
from pathlib import Path

import psycopg
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

from ragci.contract import Chunk, ParamSpec, RetrievalTrace, Step, adapter

CORPUS = Path(__file__).parent / "corpus"
DSN = os.environ.get("DATABASE_URL", "postgresql://postgres:ragci@localhost:55432/ragci")
MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions
DIM = 384


def _chunks(text: str, size: int, overlap: int = 40):
    """Fixed windows with exact offsets. A real pipeline would split on structure, but
    whatever it does, it has to report where each chunk came from."""
    step = max(size - overlap, 1)
    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        if end > start:
            yield start, end, text[start:end]
        if end == len(text):
            break


@adapter(
    index_time_params=[ParamSpec(name="chunk_size", values=[256, 512, 1024])],
    query_time_params=[ParamSpec(name="top_k", values=[5, 10])],
    primary_metric="recall@10",
)
class PgVectorRag:
    def __init__(self):
        self._embedder = TextEmbedding(model_name=MODEL)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._embedder.embed(texts)]

    def build_index(self, corpus, config: dict):
        size = config.get("chunk_size", 512)
        table = f"chunks_{size}"

        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            # Idempotent: a sweep calls build_index once per rung, so this runs again
            # with the same parameters and must not append to what is already there.
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(
                f"CREATE TABLE {table} (id bigserial PRIMARY KEY, doc_id text NOT NULL,"
                f" char_start int NOT NULL, char_end int NOT NULL, body text NOT NULL,"
                f" embedding vector({DIM}))"
            )

            rows = [
                (path.name, start, end, body)
                for path in sorted(CORPUS.glob("*.txt"))
                for start, end, body in _chunks(path.read_text(encoding="utf-8"), size)
            ]
            vectors = self._embed([r[3] for r in rows])
            with conn.cursor().copy(
                f"COPY {table} (doc_id, char_start, char_end, body, embedding) FROM STDIN"
            ) as copy:
                for (doc_id, start, end, body), vector in zip(rows, vectors, strict=True):
                    copy.write_row((doc_id, start, end, body, str(vector)))

            # Index after the load: building it first would rebuild on every insert.
            conn.execute(
                f"CREATE INDEX ON {table} USING hnsw (embedding vector_cosine_ops)"
            )
        return table

    def retrieve(self, query: str, index, config: dict) -> RetrievalTrace:
        table = index or f"chunks_{config.get('chunk_size', 512)}"
        [vector] = self._embed([query])

        with psycopg.connect(DSN) as conn:
            register_vector(conn)
            found = conn.execute(
                f"SELECT doc_id, char_start, char_end, body, 1 - (embedding <=> %s::vector)"
                f" FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (str(vector), str(vector), config.get("top_k", 10)),
            ).fetchall()

        chunks = [
            Chunk(text=body, doc_id=doc_id, char_start=start, char_end=end, score=score)
            for doc_id, start, end, body, score in found
        ]
        return RetrievalTrace(steps=[Step(query=query, chunks=chunks)], latency_ms=1.0)
