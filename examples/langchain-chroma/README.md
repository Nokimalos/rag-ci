# rag-ci against LangChain + Chroma

A working adapter for the most common stack: `RecursiveCharacterTextSplitter` for
chunking, Chroma for storage and retrieval, and Chroma's built-in ONNX embeddings
(`all-MiniLM-L6-v2`, 384 dimensions). No API key.

```bash
uvx --with chromadb --with langchain-text-splitters rag-ci run --metric recall@3
```

The first run downloads the embedding model (~80 MB) into `~/.cache/chroma`.

## The one line that matters

```python
RecursiveCharacterTextSplitter(chunk_size=..., add_start_index=True)
```

Without `add_start_index`, LangChain reports `metadata: {}` — no offsets. rag-ci then
falls back to token overlap, flags the run as `degraded_matching`, and measures less
precisely. With it, offsets survive the trip through Chroma intact (`int` type included),
and matching is exact: `degraded_matching: False`.

That was the open question about this contract, and this example is the answer.

## How the golden set was made

Not by hand. The corpus is six documents, each carrying one fact; `qa.jsonl` is the six
questions and their answers, written as anyone already has them. Then:

```bash
rag-ci golden anchor --qa qa.jsonl --corpus corpus
```

which located each answer and wrote `golden.jsonl` with character offsets. Six of six
anchored, nothing left for review — the corpus quotes its answers verbatim, which is the
easy case.

## Not run in CI

`chromadb` and `langchain-text-splitters` are not dependencies of rag-ci, and pulling a
vector store and an ONNX runtime into the test matrix to check an example is a poor
trade. `examples/reference/` is the one that gates every pull request. This one is here
to be read and copied.
