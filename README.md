# rag-ci

[![CI](https://github.com/Nokimalos/rag-ci/actions/workflows/ci.yml/badge.svg)](https://github.com/Nokimalos/rag-ci/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

Regression testing and configuration sweeps for RAG pipelines.

> **Status: usable.** Build a golden set, measure against it, gate pull requests on the
> result, sweep configurations to find what actually wins, and score answer grounding —
> including as a GitHub Action. See [the design document](docs/design.md).

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

## Commands

```bash
uvx rag-ci init            # scaffold an adapter for your pipeline   ✅
uvx rag-ci golden gen      # generate candidate questions            ✅
uvx rag-ci golden review   # accept / edit / reject, then commit     ✅
uvx rag-ci run             # measure, with confidence intervals      ✅
uvx rag-ci gate            # fail the PR only on a real regression   ✅
uvx rag-ci sweep           # find the configuration that wins        ✅
uvx rag-ci run --judge     # score answer grounding, not just recall ✅
uvx rag-ci judge calibrate # check the judge before trusting it      ✅
```

Generating questions needs the optional extra and an `ANTHROPIC_API_KEY`
(`uvx "rag-ci[generate]" golden gen …`). Everything else installs and runs with no model
dependency at all. See [docs/golden-sets.md](docs/golden-sets.md) and
[docs/sweeps.md](docs/sweeps.md), and [docs/judging.md](docs/judging.md).

## Use it in CI

```yaml
name: rag-ci
on: pull_request

permissions:
  contents: read
  pull-requests: write

jobs:
  retrieval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: Nokimalos/rag-ci@v0.4.0
        with:
          adapter: ragci_adapter.py
          golden: tests/golden.jsonl
```

The gate exits `1` on a regression and `2` when it cannot make a trustworthy comparison —
an invalid run, or a baseline recorded against a different golden set. See
[docs/github-action.md](docs/github-action.md) for recording your first baseline and tuning
`min-effect`.

A worked example lives in [`examples/reference/`](examples/reference): an adapter, its
golden set, and a committed baseline. rag-ci gates it on every pull request to this
repository.

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
