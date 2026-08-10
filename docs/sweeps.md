# Configuration sweeps

```bash
uvx rag-ci sweep --metric recall@10
```

The sweep searches the parameter grid your adapter declares and reports which
configuration wins — without evaluating every combination against every case.

> **The sweep ranks mean scores. It does not run a significance test.** Unlike `gate`,
> which blocks only on a statistically significant regression, `sweep` compares raw means
> and returns the highest. The winner may be ahead by noise, and the report says so when
> the elimination cut was arbitrary — but it does not attach a confidence interval or a
> p-value to the ranking. Treat the winner as a candidate to confirm with `run` and `gate`,
> not as a proven improvement. Correcting the ranking for multiple comparisons is designed
> (`ragci.stats.holm_bonferroni` exists and is tested) but not yet wired in.

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

**`ragci.poolcurve` is a standalone module, not part of `sweep`.** Import and call it
yourself; `rag-ci sweep` does not extrapolate, and reports only what it measured. Wiring
the two together is designed but unimplemented.

`ragci.poolcurve` measures the metric at increasing pool sizes and fits recall against
`log10(pool size)`, which is roughly how retrieval degrades as distractors accumulate. It
reports the projection to full corpus size with an uncertainty band derived by
bootstrapping the fit.

When the observed points do not follow that shape, it reports `reliable: false` rather than
a confident number. Two points are never judged reliable — a straight line always fits two
points perfectly, so the fit quality tells you nothing. Measure at three pool sizes or more.
