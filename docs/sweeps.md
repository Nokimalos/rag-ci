# Configuration sweeps

```bash
uvx rag-ci sweep --metric recall@10
```

The sweep searches the parameter grid your adapter declares and reports which
configuration wins — without evaluating every combination against every case.

## The winner has to beat the field

Ranking picks the largest mean. That is not the same question as *is this configuration
actually better*, and a sweep that only answers the first one reports coin tosses as
results.

So the final rung carries at least two configurations, evaluated on the same cases in the
same order, and the winner is tested against each of them with the paired bootstrap —
corrected with Holm-Bonferroni, because comparing one winner against nineteen runners-up
at `alpha=0.05` makes one spurious "significantly better" the expected outcome rather
than a surprise.

Two possible verdicts:

```console
Winner confirmed: ahead of all 2 finalist(s), p ≤ 0.0031 after Holm-Bonferroni.
```

```console
No clear winner: the top configuration is not statistically separable from the rest of
the final rung. Picking it over these is a preference, not a measured improvement.
  vs chunk_size=70, top_k=3: +0.167 (95% CI [0.000, 0.500], p=0.3307)
```

Tune the threshold with `--alpha`. `outcome.decisive` is `False` whenever the winner did
not separate itself, so a script can act on it without parsing the text.

## Sharing the result

```bash
uvx rag-ci sweep --report report.html
```

One self-contained page — no CDN, no fonts, no scripts, nothing fetched at render time. A
report that needs the network renders as a blank box the day the host moves.

It carries what the terminal shows and a little the terminal cannot: the verdict with each
finalist's interval and p-value, the tie-break warning when the cut was one, the projection
when there is one, and the search cost.

**Configurations eliminated earlier get their own table, below the finalists**, with the
reason attached. They were scored on fewer cases, so charting them beside the final rung
would invite a comparison the numbers do not support — and a page makes that comparison
look far more convincing than a table of numbers does.

## Reusing evaluations

A sweep re-evaluates configurations across rungs, and re-running one after changing a
single parameter re-measures everything that did not change. With an LLM judge in the loop
that is slow *and* billed.

```bash
uvx rag-ci sweep --cache
```

Measured on a four-configuration grid: a warm cache reuses all four evaluations and
rebuilds **zero** indexes, against two rebuilds cold. The cache is consulted *before* the
index is built, because on a real corpus reindexing is the dominant cost — a hit that
still reindexed would save the cheap half of the work.

### It is off by default, and that is deliberate

A cache that serves a stale result is worse than no cache, because what this tool emits is
a verdict people act on. The key covers everything that decides an evaluation: the
adapter's source, the configuration, the exact cases in order, the metrics, and the rag-ci
version — so editing your retriever, reordering the golden set, or upgrading rag-ci all
invalidate it.

What a key **cannot** cover is anything outside your adapter file: an embedding service
that changed behind the same API, an index rebuilt by another process, a vector store
mutated between runs. Nothing can detect those, which is why you opt in rather than opt
out. `rm -rf .ragci/cache` when in doubt; re-measuring is always available.

### What this costs

Carrying a runner-up to the end is one extra full-corpus evaluation. On nine
configurations and ninety cases the sweep spends 360 case-evaluations instead of 280 —
still well under the 810 an exhaustive grid would cost. That is the price of the verdict,
and a sweep that cannot tell you whether its winner won is not worth the 280 either.

### The limit worth knowing

The winner is selected and tested on the same cases, so this is post-selection inference
and the p-values lean optimistic. It answers "is the winner ahead **on these cases**",
not "will it stay ahead on new ones". The asymmetry is still useful: when even an
optimistic test finds nothing, the field really is indistinguishable — which is the case
worth catching. Testing finalists on held-out cases would remove the bias and is tracked
separately.

## What successive halving buys, and what it costs

Every configuration runs against a small sample of cases. The worst two thirds are
dropped, and survivors advance to a rung with three times as many cases. Repeat until one
remains.

On a grid of 9 configurations and 90 cases, an exhaustive search costs 810 case-evaluations;
halving costs 270. The saving grows with the grid.

**The cost is real and worth stating plainly: a configuration that only shines on hard
cases can be eliminated early**, before it ever sees one. Halving assumes early performance
predicts late performance. That assumption usually holds and occasionally does not. If you
suspect it fails for your workload, raise `--min-cases` so the first rung is large enough to
be representative, or lower `--eta` so fewer configurations are dropped per step.

## Below a certain size, the sweep is a grid search

A sweep needs roughly `eta × min-cases` cases before a second rung fits. With the defaults
(`eta 3`, `min-cases 10`) that is 30 cases; below that you get a single rung, every
configuration is evaluated on everything, and the report says `0% saved`.

That is the correct behaviour, not a limitation to work around. Eliminating a configuration
requires evidence, and a 12-case sample does not have any to spare. **A sweep over 20 cases
finds noise, not a winner** — the confidence intervals on each configuration will overlap
almost completely. Build the golden set first (see [docs/golden-sets.md](golden-sets.md)),
then sweep.

## Index-time and query-time parameters

The adapter declares which parameters force a reindex:

```python
@adapter(
    index_time_params=[ParamSpec(name="chunk_size", values=[256, 512, 1024])],
    query_time_params=[ParamSpec(name="top_k", values=[5, 10, 20])],
)
```

`top_k` varies freely against an existing index; `chunk_size` requires rebuilding it. The
sweep orders each rung so every query-time variant of one index is evaluated before the
index changes — one build per index rather than one per configuration.

On a small corpus this is invisible. On five million documents it is the difference between
a sweep that finishes and one that does not.

Adapters that cannot rebuild an index — you are evaluating a production system — simply omit
`build_index`. Restrict the sweep to what can vary:

```bash
uvx rag-ci sweep --only top_k --only rrf_weight
```

## Reading the result

```
Searched 4 configurations in 2 rung(s): 14 case-evaluations instead of 24 (42% saved).
```

The table ranks configurations by the deepest rung they reached, then by score. A
configuration eliminated in rung 0 shows the score it earned on the small sample — enough to
see why it was dropped, not enough to conclude it is bad.

**A winner is not automatically a significant winner.** When every configuration scores the
same on the first rung, the cut is settled by tie-break rather than by evidence, and the
sweep says so:

```
Warning: configurations were eliminated while tied with the survivors, so the cut was
decided by tie-break rather than by evidence. This winner is a draw, not a result.
```

That warning is common on small golden sets, where a handful of easy questions leave every
configuration on 1.000. The fix is more cases, not a different sweep.

Scores from configurations eliminated early are marked `*`: they were measured on fewer
cases and **are not comparable** with the winner's. A configuration showing 1.000 on two
cases has not beaten one showing 0.833 on six. Confirm a candidate against your baseline with `rag-ci gate`
before adopting it: that path runs the paired bootstrap and will tell you whether the
difference survives scrutiny.

## Measuring on a sub-corpus

Sweeping a full corpus of millions of documents is often impractical — every index-time
change means reindexing all of it. The usual approach is to sweep against a bounded
sub-corpus, then validate the finalists at full scale.

The catch is that a sub-corpus **overstates** recall: fewer distractors, easier retrieval.
Reporting that number as if it were the real one is the dishonest option.

Pass `--extrapolate-to N` with a `--corpus`, and the sweep measures the winner at growing
pool sizes and projects to a corpus of `N` documents:

```console
Projected to 1,000,000 documents: 0.612 (95% CI [0.548, 0.671], R²=0.97).
A projection, not a measurement.
```

**Every pool keeps the documents the golden set needs.** Only the distractors grow. Slicing
the corpus instead would remove the documents the questions are about, and the curve would
measure "the answer is no longer there" rather than "the answer is harder to find" —
recall would collapse for a reason that has nothing to do with scale.

The projection is refused in two cases, each with its own message: when the measured points
do not follow a log-linear shape (`R² < 0.9`), and when the score never moved across the
range you measured. A plateau is not a trend, and extending one to a corpus a thousand times
larger would be maximum confidence drawn from an absence of evidence.

Cost: one index build and one evaluation per pool size, on the winner only. `--pool-points`
sets how many sizes to measure (default 4, minimum 3 for a fit that can be judged).

`ragci.poolcurve` measures the metric at increasing pool sizes and fits recall against
`log10(pool size)`, which is roughly how retrieval degrades as distractors accumulate. It
reports the projection to full corpus size with an uncertainty band derived by
bootstrapping the fit.

When the observed points do not follow that shape, it reports `reliable: false` rather than
a confident number. Two points are never judged reliable — a straight line always fits two
points perfectly, so the fit quality tells you nothing. Measure at three pool sizes or more.
