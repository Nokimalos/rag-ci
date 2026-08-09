# rag-ci

Regression testing and configuration sweeps for RAG pipelines.

> **Status: early.** `rag-ci init` and `rag-ci run` work today — write an adapter, bring a
> golden set, get tier-1 metrics with confidence intervals. `gate`, `sweep`, and golden-set
> generation are next; see [the design spec](docs/superpowers/specs/2026-08-09-rag-ci-design.md).

## The problem

You changed the chunk size. Did retrieval get better?

Most teams cannot answer that. They ship a change, eyeball a few queries, and move on. The
2026 literature suggests this is expensive: *[Beyond the
Reranker](https://arxiv.org/html/2606.28367v1)* finds that many retrieval enhancements stop
contributing anything once a strong reranker is present. People are stacking techniques that
do nothing and paying for them in latency and tokens.

Existing evaluation tools do not close the gap. They report a bare score with no confidence
interval — so a recall@10 moving from 0.71 to 0.74 on 200 questions looks like progress when
it is indistinguishable from noise. None of them run as a gate on a pull request, and none
of them sweep configurations.

Meanwhile the academic side settled this years ago: paired bootstrap tests with 10,000
resamples and 95% confidence intervals are standard in
[T2-RAGBench](https://arxiv.org/html/2604.01733v1) and HetDocQA. That rigor has not reached
the tools practitioners actually run.

## What rag-ci will do

```bash
uvx rag-ci init            # scaffold an adapter for your pipeline
uvx rag-ci golden gen      # generate candidate questions from your corpus
uvx rag-ci golden review   # accept / edit / reject them, then commit the result
uvx rag-ci run             # measure
uvx rag-ci gate            # fail the PR only on a statistically real regression
uvx rag-ci sweep           # find the configuration that actually wins
```

## What makes it different

- **Ground truth anchored to document passages, never to chunks.** Change the chunk size and
  your golden set still works. This is what makes sweeping possible at all.
- **Statistics as the core mechanism.** Every metric carries a confidence interval. The gate
  uses a paired bootstrap test and blocks only when a regression is both statistically
  significant and large enough to matter.
- **Built for real corpora.** Stratified sampling for question generation, successive halving
  instead of grid search, index-time and query-time parameters separated so sweeps rebuild
  indexes as rarely as possible, and a recall-vs-pool-size curve so results measured on a
  sub-corpus are extrapolated honestly rather than quietly overstated.
- **Agnostic by design.** No RAG framework as a dependency. You write one adapter file; your
  stack stays yours. The adapter contract captures multi-step retrieval trajectories, so
  agentic and multi-hop pipelines are first-class.

## License

MIT
