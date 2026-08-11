# rag-ci — Design

Why rag-ci is built the way it is. Read this before proposing an architectural change —
most of what looks arbitrary here is load-bearing, and the reasoning is written down so it
can be argued with rather than rediscovered.

This document tracks intent, not progress. See the README for what actually ships today.

**Designed but not implemented as of 0.6.0** — listed here so nothing below reads as a
feature list: two-stage evaluation on large corpora, and gating on latency.
The last of those is not simply unbuilt: timings are deliberately excluded from the run
record so two identical runs produce identical bytes, and gating on latency would mean
reopening that.

*Last revised 2026-08-10.*

---

## 1. Problem

Teams change their RAG pipeline constantly — a new chunk size, a different embedding
model, an added reranker — and have no idea whether any of it helped. The 2026 literature
makes the point sharply: *Beyond the Reranker* (arXiv 2606.28367) finds that many retrieval
enhancements stop contributing anything once a strong reranker is present. People are
stacking techniques that do nothing, and paying for them in latency and tokens.

Existing tools do not close this gap:

- **ragas / DeepEval** produce a bare score with no confidence interval. A recall@10 that
  moves from 0.71 to 0.74 on 200 questions is indistinguishable from noise, and they will
  not tell you that.
- **promptfoo** is prompt-centric and does not model retrieval quality.
- None of them run as a **regression gate** on a pull request, and none of them **sweep
  configurations**.

Meanwhile the academic side has settled on the right statistics — paired bootstrap tests
with 10,000 resamples and 95% confidence intervals are standard in T2-RAGBench (EACL 2026)
and HetDocQA — and none of it has reached the tools practitioners actually run.

**rag-ci closes that gap:** a CLI plus a GitHub Action that answers "did this change make
retrieval better, worse, or neither — and are you sure?"

## 2. Non-goals

rag-ci does **not** ingest your corpus, host a vector store, or implement retrieval. Your
pipeline does that. rag-ci measures it.

The one place this boundary bends is the sweep, which must rebuild indexes to vary
index-time parameters. Section 8 addresses that directly.

## 3. Core design decisions

### 3.1 Ground truth is anchored to passages, never to chunks

If the golden set says "the answer is in chunk #47", then changing `chunk_size` destroys
the entire golden set and the sweep cannot exist. So a golden case references a
**document passage** — `{doc_id, char_start, char_end, text}` — and scoring asks whether a
retrieved chunk *covers* that passage.

**Coverage rule:** a retrieved chunk covers a gold passage when it has the same `doc_id`
and `|intersection| / |passage|` ≥ `coverage_threshold` (default 0.5). When the adapter
does not report character offsets, fall back to normalized token overlap against the
passage text at the same threshold, and mark the run as using degraded matching.

This is what makes the golden set survive any change of chunking, embedding model, or
retriever. It is the load-bearing decision of the whole design.

### 3.2 The adapter contract captures trajectories, not single queries

Agentic RAG — query decomposition, multi-hop, iterative retrieval — is standard in 2026. A
contract shaped like `retrieve(query) -> chunks` cannot represent it and would be obsolete
within months. The contract captures a **`RetrievalTrace`**: an ordered list of steps, each
with its sub-query and the chunks it returned. A single-turn pipeline is simply a
one-step trajectory, so nothing is harder for the common case.

### 3.3 Index-time and query-time parameters are declared separately

`top_k`, RRF fusion weights, and reranker on/off are free to vary — same index. `chunk_size`,
embedding model, Matryoshka dimension, and contextual retrieval on/off each force a
reindex. The adapter declares which is which, and the sweep orders its grid to rebuild the
index as rarely as possible. On a large corpus this is an order-of-magnitude difference in
sweep cost.

Adapters that cannot rebuild an index (evaluating an existing production system) simply
omit `build_index`; rag-ci then sweeps query-time parameters only.

### 3.4 Statistics are the core mechanism, not a feature

Every metric is reported with a 95% bootstrap confidence interval. The gate compares run
against baseline with a **paired bootstrap test** over the same questions, which is far
more sensitive than comparing two independent means.

A useful consequence: this subsumes the LLM-judge variance problem. There is no ad-hoc rule
for "the judge is too noisy to block" — judge noise widens the confidence interval, which
makes the difference non-significant, which means the gate does not block. One mechanism
governs both tiers.

## 4. Adapter contract

A user writes one file, `ragci_adapter.py`:

```python
from ragci import ParamSpec, RetrievalTrace, Step, Chunk, Answer, adapter

@adapter(
    index_time_params=[
        ParamSpec("chunk_size", values=[256, 512, 1024]),
        ParamSpec("embedding_model", values=["voyage-3", "text-embedding-3-large"]),
        ParamSpec("embedding_dim", values=[256, 1024]),          # Matryoshka truncation
        ParamSpec("contextual_retrieval", values=[True, False]),
    ],
    query_time_params=[
        ParamSpec("top_k", values=[5, 10, 20]),
        ParamSpec("rrf_weight", values=[0.3, 0.5, 0.7]),
        ParamSpec("reranker", values=[None, "cohere-rerank-3"]),
    ],
    primary_metric="recall@10",
)
class MyRag:
    def build_index(self, corpus: Corpus, config: Config) -> IndexHandle:
        """Optional. Omit to evaluate an existing index (query-time sweep only)."""

    def retrieve(self, query: str, index: IndexHandle, config: Config) -> RetrievalTrace:
        """Required."""

    def answer(self, query: str, trace: RetrievalTrace, config: Config) -> Answer:
        """Optional. Required only for tier-2 generation metrics."""
```

Core types, all Pydantic:

```python
class Chunk(BaseModel):
    text: str
    doc_id: str
    char_start: int | None = None      # enables exact coverage matching
    char_end: int | None = None
    score: float | None = None

class Step(BaseModel):
    query: str                          # the sub-query; equals the original for single-turn
    chunks: list[Chunk]

class RetrievalTrace(BaseModel):
    steps: list[Step]
    latency_ms: float
    tokens: TokenUsage | None = None
    cost_usd: float | None = None

class Answer(BaseModel):
    text: str
    cited_chunks: list[Chunk] = []
    latency_ms: float
    tokens: TokenUsage | None = None
    cost_usd: float | None = None
```

## 5. Golden set

Generated synthetically from the corpus, then reviewed by a human, then versioned in the
repository. It becomes an asset the team owns and grows — this is the durable advantage
over regenerating throwaway questions on every run.

Stored as JSONL so it streams and diffs cleanly:

```json
{
  "id": "q_0a3f21",
  "question": "What notice period applies to a fixed-term contract?",
  "required_passages": [
    {"doc_id": "handbook_v3", "char_start": 12040, "char_end": 12310, "text": "..."}
  ],
  "reference_answer": "...",
  "multi_hop": false,
  "provenance": "synthetic-reviewed",
  "reviewed_at": "2026-08-09",
  "strata": {"source": "handbook", "cluster": 12}
}
```

**Generation at scale.** A five-million-document corpus is never fed to an LLM. rag-ci
embeds a sample, clusters it, and draws a stratified sample across clusters and declared
metadata, then generates candidates from that sample. The report states coverage
explicitly: how many strata exist, how many were sampled, how many produced accepted cases.

**Review.** `rag-ci golden review` walks candidates in a `rich` terminal flow —
accept / edit / reject — ordered by the generator's self-reported confidence, lowest first,
so the cases most likely to be wrong are seen while attention is fresh. Review state
persists, so the command is resumable across sessions.

## 6. Metrics

**Tier 1 — deterministic, free, blocking.** Computed over the union of chunks across all
trajectory steps:

- `recall@k`, `precision@k` — passage coverage
- `all_passages_recall@k` — fraction of cases where *every* required passage was covered.
  The metric that matters for multi-hop; the plain mean hides partial failures.
- `MRR`, `nDCG@k` — ranked at the first chunk covering a required passage
- Operational: latency p50/p95, cost per query, tokens per query

**Tier 2 — LLM judge, opt-in, non-blocking by default.** Requires `answer()`:

- `faithfulness` — is every claim grounded in the retrieved chunks?
- `citation_accuracy` — do cited chunks actually support the cited statement?

Judge runs at temperature 0 with a **single vote per case**. Repeated votes at temperature 0
would be pure waste — they return the same answer. The uncertainty that matters when
comparing two systems is variance *across questions*, and the paired bootstrap already
captures it by resampling cases.

Judge stability is a separate concern with a separate command: `rag-ci judge calibrate`
re-scores a sample under perturbation (permuted chunk order, paraphrased rubric) and
reports flip rate. A judge that flips on chunk ordering is not fit to gate anything, and
the user should learn that once — not pay for it on every run.

Tier 2 becomes blocking only if the user opts in, and even then the paired bootstrap
governs, so a noisy judge cannot produce a failure on its own.

## 7. The gate

```
rag-ci gate --baseline .ragci/baseline.json
```

Blocks the PR when **both** conditions hold on the primary metric:

1. The paired bootstrap test (10,000 resamples, one-sided, α = 0.05) rejects the null.
2. The point estimate degrades by more than `min_effect` (default 0.02).

Requiring both prevents the two failure modes that kill adoption: blocking on noise, and
blocking on a statistically real but practically irrelevant 0.3% drop.

Additional rules:

- **No baseline?** The first run establishes it and never blocks.
- **Golden set changed?** Each run stores the golden set hash. A mismatch invalidates the
  comparison and the gate reports "baseline stale" rather than a bogus verdict.
- **Run invalid?** (see §9) The gate reports an infrastructure failure, distinct from a
  quality regression.

Output: a console table via `rich`, a Markdown report for the PR comment, and
`.ragci/run.json` for machines.

## 8. Sweep, and scale

### 8.1 Successive halving, not grid search

A full grid of 5 parameters × 4 values against 5,000 cases is 5M evaluations. Instead:
all configurations run against a small case sample (rung 0), the worst `1 - 1/η` are
eliminated (η = 3), survivors advance to a rung with η× more cases. Final ranking applies
Holm-Bonferroni correction for multiple comparisons.

Within each rung, configurations are ordered by their index-time parameters so that all
query-time variants of one index are evaluated before the index is rebuilt.

### 8.2 Two-stage evaluation on large corpora

Reindexing five million documents even five times costs hours and GPU budget. So:

- **Stage 1 — exploration.** Successive halving runs against a bounded sub-corpus: every
  document containing a gold passage, plus a stratified sample of distractors (default
  100k documents).
- **Stage 2 — validation.** The top 2–3 finalists are re-evaluated against the full corpus.

### 8.3 The honesty mechanism: recall vs. pool size

A sub-corpus overstates recall — fewer distractors, easier retrieval. Reporting stage-1
numbers as if they were real would make the tool misleading.

So rag-ci measures the primary metric at increasing distractor pool sizes (1k, 10k, 100k,
and full where affordable). Recall degrades approximately linearly in `log(pool_size)`, so
the report fits a log-linear model, extrapolates to full corpus size, and derives the
uncertainty band by bootstrapping the fit. When the observed points deviate from log-linear
beyond tolerance, the report says the extrapolation is unreliable rather than printing a
confident wrong number.

This is the piece no existing tool has, and it is what makes a proxy measurement
defensible rather than convenient.

### 8.4 Data handling

Nothing is held in memory that scales with the corpus: a `doc_id → (path, offset)` index on
disk, streamed JSONL for the golden set and run records, an async runner with bounded
concurrency, and a content-addressed disk cache keyed on `hash(model + params + input)`.
The cache is what makes re-runs and duplicate sweep configurations nearly free.

## 9. Error handling

**A failure must never be recorded as a score of zero.** Otherwise a rate-limited afternoon
looks exactly like a quality regression.

- A case that raises inside the adapter gets status `error`: excluded from metrics,
  counted, and displayed. Above 5% errors the run is marked **invalid** — not bad.
- Rate limits and timeouts: exponential backoff with jitter, bounded attempts.
- `--max-cost` stops cleanly; the partial report is marked incomplete and is never compared
  against a baseline.
- **Determinism:** with a warm cache and a temperature-0 judge, re-running an unchanged run
  reproduces every retrieved chunk, score, and metric exactly. Timing fields are inherently
  variable, so they live in a separate `timings` block that is excluded from the determinism
  check and from run-to-run diffs — but still reported and still gated on p95. This is a
  tested property, not an aspiration.

## 10. Modules

| Module | Responsibility |
|---|---|
| `contract` | Adapter protocol, `ParamSpec`, core Pydantic types |
| `corpus` | On-disk document index, stratified sampling, sub-corpus construction |
| `golden` | Synthetic generation, interactive review, JSONL persistence |
| `runner` | Executes cases against a retriever, collects traces, async with bounded concurrency |
| `metrics` | Tier-1 metrics via passage coverage matching |
| `stats` | Bootstrap CIs, paired bootstrap tests, Holm-Bonferroni |
| `judge` | Tier-2 LLM metrics with per-case variance |
| `baseline` | Run/baseline comparison, gate decision |
| `sweep` | Successive halving, index-time ordering, pool-size curve |
| `report` | Console (rich), Markdown, JSON |
| `cache` | Content-addressed disk cache |
| `cli` | Typer orchestration |

Each is testable in isolation. `metrics` and `stats` are deliberately split: statistical
machinery must be verifiable against known distributions without any retrieval involved.

## 11. Testing

TDD throughout. A reference in-memory adapter — toy corpus, trivial BM25 retriever — lets
the entire pipeline be tested with no network and no API key.

Non-negotiable tests:

- **Metrics against hand-computed values** on five-document cases. A wrong metric
  invalidates the entire tool.
- **Bootstrap against known distributions**: null effect → not significant; strong effect →
  significant; false-positive rate within tolerance across repeated trials.
- **Passage matching edge cases**: partial overlap, passage straddling two chunks, same
  offsets in a different document, missing offsets triggering the fallback.
- **Cache reproducibility**: identical re-run produces identical bytes.
- Markdown report snapshots; recorded cassettes for real LLM calls, replayed in CI.

## 12. Stack and distribution

Python 3.12 with uv. Typer for the CLI, Pydantic for contracts and config, `rich` for
console and review UI, NumPy for the bootstrap. **No RAG framework as a dependency** — being
agnostic is the entire point.

Published to PyPI as `rag-ci` (name verified available 2026-08-09), runnable as
`uvx rag-ci`. A composite GitHub Action lives in the same repository.

## 13. v1 scope

**Ships:** `init`, `golden gen`, `golden review`, `run`, `gate`, `sweep`, `report` · full
tier 1 · tier 2 on Anthropic only behind an open interface · GitHub Action · reference
adapter on a public corpus · documentation.

**Explicitly out:** web UI or hosted dashboard, historical run database, ragas/DeepEval
import, multi-provider judge, GitLab/Jenkins support, generation metrics beyond
faithfulness and citation accuracy.

Each excluded item is roughly a week of work that proves nothing the included set does not
already prove.
