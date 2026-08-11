# rag-ci

[![CI](https://github.com/Nokimalos/rag-ci/actions/workflows/ci.yml/badge.svg)](https://github.com/Nokimalos/rag-ci/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

**Stop shipping RAG regressions.**

rag-ci measures every change to your RAG pipeline against a baseline and fails the pull
request when retrieval actually got worse — not when it merely looks worse.

See it catch one, in one command and under a second — no API key, nothing to download:

```console
$ uvx rag-ci demo
Built a synthetic corpus: 144 documents, 144 questions anchored to character offsets.

1. Baseline — chunk_size=240
┏━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ metric     ┃  mean ┃         95% CI ┃
┡━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ recall@3 * │ 0.951 │ [0.917, 0.986] │
└────────────┴───────┴────────────────┘

2. Someone halves the chunk size to fit more of them in context — chunk_size=120
┏━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ metric     ┃  mean ┃         95% CI ┃
┡━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ recall@3 * │ 0.736 │ [0.667, 0.806] │
└────────────┴───────┴────────────────┘

3. The gate compares them over the same questions
REGRESSION — the gate fails — recall@3 dropped by 0.215
(95% CI [-0.292, -0.139], p=0.0000) over 144 paired cases.
```

Exit code 1. In CI, that is a blocked pull request.

**Nothing there is a recording.** The corpus is generated on your machine and measured on
the spot, through the same code paths `run` and `gate` use — a canned transcript would
contradict the one thing this project argues for. Reproduce it yourself; the numbers will
match, because they are deterministic, not because they were typed in.

## Why the interval is the point

The demo above blocks because 144 questions are enough to be sure. The opposite case
matters just as much. In [`examples/reference/`](examples/reference) — six questions —
cutting `top_k` to 1 drops recall@3 from 0.833 to 0.667, and rag-ci does **not** block:

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
uvx rag-ci demo            # watch a regression get caught, in one second ✅
uvx rag-ci init            # scaffold an adapter for your pipeline   ✅
uvx rag-ci golden anchor   # anchor a Q/A set you already have        ✅
uvx rag-ci golden gen      # generate candidate questions            ✅
uvx rag-ci golden review   # accept / edit / reject, then commit     ✅
uvx rag-ci run             # measure, with confidence intervals      ✅
uvx rag-ci gate            # fail the PR only on a real regression   ✅
uvx rag-ci sweep --report  # a shareable HTML page of the result       ✅
uvx rag-ci sweep --cache   # find the configuration that provably wins ✅
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
      - uses: Nokimalos/rag-ci@v0.7.1
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

Two worked examples. [`examples/reference/`](examples/reference) is dependency-free and
gates every pull request to this repository.
Two more are real pipelines:
[`examples/langchain-chroma/`](examples/langchain-chroma) shows the one flag
(`add_start_index=True`) that decides whether rag-ci matches passages exactly or falls
back to token overlap, and [`examples/pgvector/`](examples/pgvector) does the same in
plain `psycopg` against Postgres — where the requirement is just two integer columns
beside the vector.

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
