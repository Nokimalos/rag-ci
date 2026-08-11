# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version stays below 1.0, the adapter contract may change in a minor release.

## [Unreleased]

## [0.7.1] — 2026-08-11

Both fixes came from running rag-ci over SQuAD — 48 Wikipedia articles, 300 questions,
LangChain chunking, Chroma storage — rather than from reading our own code.

### Fixed

- **A tie-break pointed at the wrong remedy.** When a rung eliminated configurations that
  were tied, rag-ci said "add cases and sweep again". On a golden set of 300 cases whose
  first rung used 10, that sends you to collect questions you already have — the evidence
  was there, the rung just did not use it. The message now names `--min-cases` when the
  first rung is small relative to the deepest, and keeps the original wording when the run
  genuinely used everything available.
- **`examples/langchain-chroma` did not survive a real corpus**, and it ships in the sdist.
  Chroma caps a single `add()` at 5461 records, which 1.6 MB at `chunk_size=256` exceeds;
  and `EphemeralClient()` returns the same in-process client, so a second `build_index` for
  the same parameters found its collection already created. Worth stating in the contract:
  a sweep calls `build_index` once per rung, so it has to be idempotent.

## [0.7.0] — 2026-08-11

### Fixed

- **`rag-ci run` never called `build_index`.** An adapter that declared one — which the
  design document encourages — received `index=None` and failed with
  `AttributeError: 'NoneType' object has no attribute 'query'`, an error pointing at the
  adapter rather than at rag-ci. Building the index inside `retrieve()` instead hit the
  next layer: the runner uses eight threads, so eight vector-store clients were
  constructed at once. `run` now builds the index once, before the cases, exactly as
  `sweep` always has. Adapters that evaluate an existing production index still omit
  `build_index` and are unaffected.
- The adapter template said nothing about `build_index` or about what `index` is. It now
  does, including that a client must not be constructed inside `retrieve()`.

### Added

- **`rag-ci run --max-cost <USD>`** stops before starting further paid work once the spend
  reported by your adapter reaches the budget — contributed by @averyquinnhq. A truncated
  run is marked *incomplete*, which is distinct from *invalid*: the pipeline was healthy,
  the sample simply is not finished. The gate refuses it and `--update-baseline` refuses
  it, because half a golden set produces a plausible-looking score that means nothing
  beside a full baseline. Note that a budgeted run evaluates cases one at a time — cost is
  only known after a call returns — so it is measurably slower than an unbudgeted one.
- **`examples/langchain-chroma/`** — the first adapter in this repository written against
  a stack we did not design: LangChain chunking, Chroma storage, ONNX embeddings, no API
  key. Its golden set was produced by `golden anchor` rather than by hand.

### Answered

- **Do character offsets survive a real vector pipeline?** Yes. LangChain's splitter
  reports no offsets by default, but with `add_start_index=True` they pass through Chroma
  intact, `int` type included, and rag-ci matches passages exactly rather than falling
  back to token overlap. The friction documented since 0.4.2 is a one-flag problem, not a
  structural one.

## [0.6.0] — 2026-08-10

Everything here is additive: nothing changes for a caller who does not pass a new flag.

### Added

- **`rag-ci golden anchor`** turns an existing question/answer set into passage-anchored
  cases, so a golden set no longer has to start from scratch with an API key. Matching
  tolerates the whitespace and casing a copy-paste picks up; the offsets it reports address
  the document exactly as stored. It never guesses — an answer found in one place becomes a
  case, and anything ambiguous or absent goes to `golden.unresolved.jsonl` for a human. A
  confidently wrong offset teaches the gate to reward the wrong retrieval, and nothing
  downstream can detect that.
- **`rag-ci sweep --cache`** reuses evaluations across runs. Consulted *before* the index is
  built, since reindexing is the dominant cost of a sweep. Off by default: the key covers
  the adapter source, configuration, cases, metrics and rag-ci version, but no key can cover
  an embedding service that changed behind the same API.
- **`rag-ci run --retries N`** retries a failing case, so a network blip does not push a long
  run past its 5% error threshold. Recovery is reported rather than hidden — a run that
  needed several attempts per case says so.
- **`rag-ci sweep --report report.html`** writes one self-contained page: no CDN, no scripts,
  nothing fetched at render time. Configurations eliminated on earlier rungs get their own
  table rather than a bar beside the finalists, because they were scored on fewer cases.
- **`rag-ci sweep --holdout FRACTION`** tests the winner on cases the search never saw,
  closing the post-selection bias documented in 0.5.0. The search still picks the winner;
  re-picking it on the holdout would reintroduce the bias. A holdout reserving fewer than
  30 cases is refused rather than honoured.

### Changed

- `docs/design.md` no longer lists the cache and retries as unbuilt, and explains why gating
  on latency is a design question rather than a task: timings are deliberately excluded from
  the run record so two identical runs produce identical bytes.


## [0.5.0] — 2026-08-10

### Fixed

- **Every pull request comment reported `latency p50 0 ms`.** `save_json` excludes timings so
  two identical runs produce byte-identical output, but `Timings` defaulted to `0.0` — making
  a record read back off disk indistinguishable from a genuinely instant run. `gate` always
  reads off disk, so the figure was wrong on every comment it ever posted.
- **Five claims the code did not support**, including a comparison against promptfoo that was
  factually wrong: promptfoo does document CI/CD integration. The narrower true claim is that
  nobody else conditions the build verdict on a significance test.

### Added

- **`rag-ci demo`** runs a real regression end to end in under a second — 144 documents, 144
  questions, no API key, nothing downloaded. Generated and measured on the spot rather than
  recorded, because a canned transcript would contradict what this project argues for.
- **`sweep` tests its winner against the field** with a paired bootstrap and Holm-Bonferroni
  correction, instead of returning the largest mean, and says plainly when the field is
  statistically indistinguishable. The final rung now carries at least two configurations —
  a lone survivor has nobody to be compared against.
- **`sweep --extrapolate-to N`** projects a sub-corpus result to full scale. Every measured
  pool keeps the documents the golden set needs, so the curve measures added distractors
  rather than removed answers, and it refuses to project from points that do not support it.
- `--alpha`, `--corpus` and `--pool-points` on `sweep`; GitLab CI documentation, where
  `allow_failure: exit_codes` maps the gate's 1/2 split onto the pipeline itself.

### Changed

- **Question generation streams the corpus**, holding only identifiers between passes:
  1.3 MB peak instead of 15.4 MB on 20,000 documents, and the gap widens with document size.
- `successive_halving`'s evaluator returns per-case scores rather than a mean — ranking needs
  the mean, testing the winner needs the pairs.
- A flat pool curve is no longer treated as a perfect fit. Identical scores at every size
  leave nothing for a line to explain, and projecting a plateau with a zero-width interval
  would be maximum confidence drawn from an absence of evidence.

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
