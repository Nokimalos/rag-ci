# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version stays below 1.0, the adapter contract may change in a minor release.

## [Unreleased]

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

[Unreleased]: https://github.com/Nokimalos/rag-ci/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Nokimalos/rag-ci/releases/tag/v0.1.0
