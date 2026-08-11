# rag-ci against Postgres + pgvector

Plain `psycopg`, no framework in between — so this transfers whether you reach pgvector
through SQLAlchemy, LangChain, or raw SQL. Embeddings come from `fastembed`
(`BAAI/bge-small-en-v1.5`, 384 dimensions, ONNX, no API key).

```bash
docker run -d --name ragci-pg -e POSTGRES_PASSWORD=ragci -e POSTGRES_DB=ragci \
  -p 55432:5432 pgvector/pgvector:pg17

uvx --with "psycopg[binary]" --with pgvector --with fastembed \
  rag-ci run --metric recall@10 --top-k 10
```

Point `DATABASE_URL` at your own instance to use it instead.

## Where the offsets live

Two integer columns beside the vector:

```sql
CREATE TABLE chunks_512 (
    id bigserial PRIMARY KEY,
    doc_id text NOT NULL,
    char_start int NOT NULL,     -- these two are what rag-ci needs
    char_end   int NOT NULL,
    body text NOT NULL,
    embedding vector(384)
)
```

That is the whole integration requirement. Carry them through and rag-ci matches passages
exactly (`degraded_matching: False`); drop them and it falls back to token overlap, which
still works but measures less precisely.

## Two things that are easy to get wrong

**`build_index` must be idempotent.** A sweep calls it once per rung, so it drops and
recreates the table rather than inserting. Without that, the second call appends to the
first and every score after it is measured against a doubled corpus.

**Create the HNSW index after loading, not before.** Building it first makes it rebuild on
every insert. On a corpus of any size that is the difference between seconds and minutes.

## Verified at scale

Beyond the six documents here, this adapter was run over 48 Wikipedia articles (1.6 MB,
3417 chunks) with 300 questions from SQuAD: `recall@10 = 0.913` [0.880, 0.943], no errors,
exact offset matching throughout.

Those numbers are **not** comparable with the `langchain-chroma` example's — different
embedding model, different chunking. Comparing two pipelines by their headline score is
the mistake rag-ci exists to prevent; to compare them properly, run both against the same
golden set and let `gate` decide.

## Not run in CI

This needs a running Postgres, which is a poor trade for checking an example.
`examples/reference/` is the one that gates every pull request.
