# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version stays below 1.0, the adapter contract may change in a minor release.

## [Unreleased]

## [0.4.2] — 2026-08-09

### Fixed

- **Async adapters work.** `retrieve()`, `answer()`, and `build_index()` may be
  `async def`; rag-ci awaits them. Previously a coroutine function was handed to
  `asyncio.to_thread`, which returned the coroutine unawaited and failed several frames
  later inside Pydantic — with no indication that the adapter was async. Most modern RAG
  stacks are async, so this was the single largest barrier to adopting rag-ci.

### Known limitations

- **Character offsets are required** for exact passage matching. Pipelines that discard
  them at chunking time fall back to token overlap, which is flagged but measures less
  precisely — adopting rag-ci can mean reindexing.
- **`build_index` assumes programmatic reindexing.** Ingestion driven by a separate script
  or orchestrator puts index-time sweeps out of reach.


## [0.4.1] — 2026-08-09

### Fixed

- **The sweep no longer presents a tie-break as a result.** Found by pointing rag-ci at a
  real pipeline for the first time: all eight configurations scored identically on the
  first rung, so the cut was settled alphabetically, and the reported winner then scored
  *below* six configurations it had supposedly beaten. The outcome now records whether any
  cut fell inside a group of tied scores, and the report says so plainly instead of
  printing a confident arrow next to a coin toss.
- **Scores measured on fewer cases are marked as not comparable.** A configuration showing
  1.000 on two cases has not beaten one showing 0.833 on six, and the table no longer
  implies otherwise.


## [0.4.0] — 2026-08-09

Tier 2: answer grounding, and a way to check the judge measuring it.

### Added

- **`rag-ci run --judge`** — scores **faithfulness** (the fraction of the answer's factual
  claims the retrieved passages actually support) and **citation accuracy** (the fraction of
  the answer's citations that did any work). Per claim rather than a single verdict, so the
  run record can name the sentence that was invented.
- **`rag-ci judge calibrate`** — judges a sample twice, once with the passage order
  reversed, and reports the flip rate. Exits non-zero above the threshold. A judge whose
  verdict depends on input order is measuring position, not grounding, and its scores
  should not gate anything.
- **`docs/judging.md`**, including the cost (one model call per case) and why judge noise
  can only hide a regression, never manufacture one.

### Notes

- Requires an adapter implementing `answer()` and the optional extra. Retrieval-only
  adapters are told so rather than scored as zeros.
- Faithfulness and citation accuracy are `None`, not `0.0`, when there is nothing to
  measure — an answer asserting nothing is not unfaithful, and one citing nothing has no
  citation accuracy.
- A judge outage leaves tier-1 metrics intact and the run valid.


## [0.3.0] — 2026-08-09

Configuration sweeps. The design document is now fully delivered.

### Added

- **`rag-ci sweep`** — searches the parameter grid the adapter declares using successive
  halving: every configuration runs against a small case sample, the worst are dropped,
  survivors advance to a larger sample. The report states how many case-evaluations were
  spent against the exhaustive cost, so the saving is visible rather than implied.
- **Index-aware ordering** — each rung is ordered so every query-time variant of one index
  is evaluated before the index is rebuilt. One build per index instead of one per
  configuration, which is the difference between a sweep that finishes on a large corpus
  and one that does not.
- **`ragci.poolcurve`** — fits a metric against `log10(pool size)` and projects a
  sub-corpus measurement to full corpus size, with a bootstrapped uncertainty band. A poor
  fit reports `reliable: false` instead of a confident number, and two points are never
  judged reliable.
- **`holm_bonferroni`** in `ragci.stats` — controls the family-wise error rate when
  comparing a winner against every runner-up.
- **`docs/sweeps.md`**, including the honest caveats: a configuration that only shines on
  hard cases can be eliminated early, and below roughly `eta × min-cases` the sweep
  degenerates into an exhaustive grid search.


## [0.2.0] — 2026-08-09

Golden sets no longer have to be written by hand.

### Added

- **`rag-ci golden gen`** — samples a corpus, splits documents into passages, and generates
  a candidate question per passage. Sampling is stratified: every source is represented
  before the remaining slots are distributed proportionally, so a handful of policy
  documents among thousands of tickets is never sampled out of existence. Reads a directory
  of `.txt`/`.md` files or a JSONL export, and reports its coverage.
- **`rag-ci golden review`** — accept, edit, reject, or skip candidates in the terminal,
  ordered by lowest generator confidence first. Accepted work is flushed immediately,
  review resumes where it stopped, and rejections are remembered.
- **`rag-ci --version`**.
- **`docs/golden-sets.md`** and a contributing guide.

### Notes

- Question generation needs the optional extra: `pip install "rag-ci[generate]"` plus an
  `ANTHROPIC_API_KEY`. The core package still has **no model dependency** — `init`, `run`,
  and `gate` install and work with no API access at all.
- Default generation model: `claude-opus-5`, overridable with `--model`.

### Fixed

- The missing-extra error told users to run `pip install "rag-ci"` — `rich` was parsing
  `[generate]` as a markup tag and swallowing it, so the hint omitted the very extra it was
  telling them to install.

### Known limitations

- Configuration sweeps are not implemented yet.
- Generation metrics (faithfulness, citation accuracy) are not implemented yet — only
  retrieval is measured.

## [0.1.0] — 2026-08-09

First release. Measures retrieval quality and gates pull requests on it.

### Added

- **`rag-ci init`** — scaffolds an adapter for your pipeline.
- **`rag-ci run`** — executes a golden set against your adapter and reports tier-1 metrics
  (`recall@k`, `all_passages_recall@k`, `precision@k`, `MRR`, `nDCG@k`) with 95% bootstrap
  confidence intervals, plus latency, cost, and token counts.
- **`rag-ci gate`** — compares a run against a baseline with a paired bootstrap test and
  fails only when a regression is both statistically significant and larger than
  `--min-effect`. Exits `1` on a regression and `2` when the comparison cannot be trusted.
- **Composite GitHub Action** — measures, gates, and comments on the pull request.
- **Adapter contract** capturing multi-step retrieval trajectories, so agentic and
  multi-hop pipelines are first-class. Parameters are declared as index-time or query-time
  in preparation for sweeps.
- **Passage-anchored ground truth** — golden cases reference document spans rather than
  chunk identifiers, so changing the chunking strategy never invalidates a golden set.

### Known limitations

- Golden sets must be written by hand; generation and review arrive in a later release.
- Configuration sweeps are not implemented yet.
- Generation metrics (faithfulness, citation accuracy) are not implemented yet — only
  retrieval is measured.

[Unreleased]: https://github.com/Nokimalos/rag-ci/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/Nokimalos/rag-ci/releases/tag/v0.4.2
[0.4.1]: https://github.com/Nokimalos/rag-ci/releases/tag/v0.4.1
[0.4.0]: https://github.com/Nokimalos/rag-ci/releases/tag/v0.4.0
[0.3.0]: https://github.com/Nokimalos/rag-ci/releases/tag/v0.3.0
[0.2.0]: https://github.com/Nokimalos/rag-ci/releases/tag/v0.2.0
[0.1.0]: https://github.com/Nokimalos/rag-ci/releases/tag/v0.1.0
