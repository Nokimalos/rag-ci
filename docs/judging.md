# Judging answers (tier 2)

Tier 1 measures whether the right passages were retrieved. Tier 2 measures whether the
answer built from them is actually grounded in them.

```bash
uvx "rag-ci[generate]" run --judge
```

Requires an adapter implementing `answer()`, plus an `ANTHROPIC_API_KEY`. Retrieval-only
adapters have nothing for tier 2 to score, and `--judge` says so rather than reporting
zeros.

## What is measured

**Faithfulness** — the judge breaks the answer into factual claims and marks each as
supported or not by the retrieved passages. The score is the supported fraction.

Per claim rather than a single verdict, because knowing *which* sentence was invented is
what makes the number actionable. The run record keeps the unsupported claims so you can
read them back.

**Citation accuracy** — the fraction of the answer's citations that actually support a
claim. An answer citing five passages where one did the work scores 0.2.

Both are `None` rather than a number when there is nothing to measure: an answer that
asserts nothing has no faithfulness, and an answer that cites nothing has no citation
accuracy. Scoring an honest "I don't know" as 0.0 would punish abstention exactly as
harshly as a hallucination.

## Calibrate before you trust

**A judge is a measuring instrument, and an uncalibrated instrument produces numbers, not
measurements.**

```bash
uvx "rag-ci[generate]" judge calibrate --samples 20
```

Each sample is judged twice — once as-is, once with the retrieved passages in reverse
order. A verdict that changes is a flip. Above 10% (configurable), the command exits
non-zero and tells you not to gate on the judge's scores.

The reasoning: reversing the passage order changes nothing about whether a claim is
grounded. A judge that changes its mind is responding to position, not to content. Its
scores will drift for reasons unrelated to your pipeline, and a gate built on them will
fire at random.

Two calls per sample, once, against a lifetime of runs — worth it.

## Cost

One model call per case, on top of whatever your pipeline already spends. On a 200-case
golden set that is 200 judge calls per run.

Tier 1 stays free and deterministic, which is why it is what gates by default. Reach for
`--judge` when you are investigating answer quality, not on every commit.

`--judge-model` accepts any Claude model id; the default is `claude-opus-5`.

## Gating on tier 2

Tier-2 scores flow through the same machinery as tier 1: same bootstrap, same confidence
intervals, same paired comparison in `rag-ci gate`. You can gate on `faithfulness` by
passing it as the primary metric.

Before doing that, consider what the paired bootstrap already does for you. A noisy judge
produces noisy per-case scores, which widen the confidence interval, which makes
differences non-significant, which means the gate does not fire. **Judge noise cannot
manufacture a failure on its own** — it can only hide a real regression. That is the safe
direction for the failure mode to point in, but it also means a badly calibrated judge
turns your gate into a no-op without telling you.

Calibrate, then gate.

## When the judge is unavailable

A judge that errors or returns nothing leaves the case unjudged. Tier-1 metrics are
unaffected and the run stays valid — a judge outage is not a retrieval regression, and
conflating the two would invalidate a perfectly good measurement.
