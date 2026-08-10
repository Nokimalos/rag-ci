# rag-ci

[![CI](https://github.com/Nokimalos/rag-ci/actions/workflows/ci.yml/badge.svg)](https://github.com/Nokimalos/rag-ci/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

**Stop shipping RAG regressions.**

rag-ci measures every change to your RAG pipeline against a baseline and fails the pull
request when retrieval actually got worse — not when it merely looks worse.

```console
$ uvx rag-ci run
     rag-ci run  (6 scored cases)
┏━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ metric     ┃  mean ┃         95% CI ┃
┡━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ recall@3 * │ 0.833 │ [0.500, 1.000] │
└────────────┴───────┴────────────────┘
latency p50 1 ms  p95 1 ms

$ uvx rag-ci gate --baseline baseline.json
Gate: Pass — recall@3 moved by +0.000 (95% CI [0.000, 0.000]) over 6 paired cases.
```

That is unedited output from [`examples/reference/`](examples/reference), which you can run
yourself in two commands. Read the interval, not the mean: six questions pin recall@3 no
tighter than somewhere between 0.500 and 1.000. Every other evaluation tool would have
handed you `0.833` and let you believe it.

## Why the interval is the point

Cut `top_k` to 1 in that same example and recall@3 falls from 0.833 to 0.667. rag-ci does
not block:

```console
Gate: Pass — recall@3 moved by -0.167 (95% CI [-0.500, 0.000]) over 6 paired cases.
```

A drop that large is probably real, and rag-ci still refuses to call it — because across
six paired cases the paired bootstrap cannot separate it from noise. The interval, spanning
`-0.500` to `0.000`, is the tool telling you how little evidence six questions buy. Widen
the golden set and that interval tightens until the same change trips the gate. **A tool
that fails builds on noise gets switched off within a week**, so this one errs toward
silence and shows its evidence either way.

The same logic runs in reverse, which is where the money is: *[Beyond the
Reranker](https://arxiv.org/html/2606.28367v1)* finds that many retrieval enhancements stop
contributing anything once a strong reranker is present. Teams are stacking techniques that
do nothing and paying for them in latency and tokens, because a bare score moving from 0.71
to 0.74 reads like progress.

The academic side settled this years ago — paired bootstrap tests with 10,000 resamples and
95% confidence intervals are standard in
[T2-RAGBench](https://arxiv.org/html/2604.01733v1) and HetDocQA. That rigor had not reached
the tools practitioners actually run. rag-ci is that rigor behind a CLI and a GitHub Action.

To be precise about the gap: [promptfoo](https://www.promptfoo.dev/docs/guides/evaluate-rag/)
evaluates RAG and [runs in CI](https://www.promptfoo.dev/docs/integrations/ci-cd/), and ragas
and DeepEval both compute retrieval metrics. What none of them do is condition the build
verdict on a significance test. They compare a score to a threshold; rag-ci compares two runs
over the same questions and reports how confident the difference is.

## Commands

Every command below works today; see [the design document](docs/design.md) for how they fit
together.

```bash
uvx rag-ci init            # scaffold an adapter for your pipeline   ✅
uvx rag-ci golden gen      # generate candidate questions            ✅
uvx rag-ci golden review   # accept / edit / reject, then commit     ✅
uvx rag-ci run             # measure, with confidence intervals      ✅
uvx rag-ci gate            # fail the PR only on a real regression   ✅
uvx rag-ci sweep           # find the configuration that provably wins ✅
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
      - uses: Nokimalos/rag-ci@v0.4.2
        with:
          adapter: ragci_adapter.py
          golden: tests/golden.jsonl
```

The gate exits `1` on a regression and `2` when it cannot make a trustworthy comparison —
an invalid run, or a baseline recorded against a different golden set. See
[docs/github-action.md](docs/github-action.md) for recording your first baseline and tuning
`min-effect`.

Not on GitHub? The gate is a CLI, so any runner works. GitLab gets a one-job setup in
[docs/gitlab-ci.md](docs/gitlab-ci.md), where `allow_failure: exit_codes` maps the `1` / `2`
split onto the pipeline itself — a distinction GitHub Actions cannot express.

A worked example lives in [`examples/reference/`](examples/reference): an adapter, its
golden set, and a committed baseline. rag-ci gates it on every pull request to this
repository.

## What makes it different

- **Ground truth anchored to document passages, never to chunks.** Change the chunk size and
  your golden set still works. This is what makes sweeping possible at all.
- **A winner has to beat the field.** `sweep` does not just rank means: it carries the
  finalists to the last rung, tests the winner against each with a paired bootstrap, and
  says plainly when the field is indistinguishable instead of naming a coin toss.
- **A two-condition gate.** A regression blocks the build only when it is both statistically
  significant and larger than `min-effect`. Significance alone would block on trivia;
  effect size alone would block on noise.
- **Built for large corpora.** Question generation streams the corpus and holds only
  identifiers while it decides what to sample, never the text. Sweeps use successive halving
  instead of grid search, separate index-time from query-time parameters so indexes are
  rebuilt as rarely as possible, and `--extrapolate-to` projects a sub-corpus result to full
  scale — refusing to project when the measured points do not support it.
- **Agnostic by design.** No RAG framework as a dependency. You write one adapter file; your
  stack stays yours. The adapter contract captures multi-step retrieval trajectories, so
  agentic and multi-hop pipelines are first-class.

## License

MIT
