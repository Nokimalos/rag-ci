# Using rag-ci in GitHub Actions

On GitLab, see [docs/gitlab-ci.md](gitlab-ci.md) — the sections below on baselines,
`min-effect`, pairing and golden-set changes apply there unchanged.

Add this to `.github/workflows/rag-ci.yml`:

```yaml
name: rag-ci
on: pull_request

permissions:
  contents: read
  pull-requests: write   # only needed for the PR comment

jobs:
  retrieval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: Nokimalos/rag-ci@v0.6.0
        with:
          adapter: ragci_adapter.py
          golden: tests/golden.jsonl
```

## Recording the first baseline

The gate has nothing to compare against until a baseline exists. Run it once locally and
commit the result:

```bash
uv run rag-ci run --adapter ragci_adapter.py --golden tests/golden.jsonl
uv run rag-ci gate --update-baseline
git add .ragci/baseline.json && git commit -m "chore: record rag-ci baseline"
```

Without a baseline the action still runs and reports numbers — it just cannot fail.

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `adapter` | `ragci_adapter.py` | Your adapter file |
| `golden` | `golden.jsonl` | The golden set |
| `baseline` | `.ragci/baseline.json` | Committed baseline |
| `metric` | adapter's `primary_metric` | Metric to gate on |
| `min-effect` | `0.02` | Smallest drop worth failing over |
| `comment` | `true` | Post the report as a PR comment |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Pass — no regression, or no baseline to compare against |
| `1` | Regression — the drop is statistically significant **and** larger than `min-effect` |
| `2` | The comparison could not be made — invalid run, or stale baseline |

Code `2` is deliberately distinct from `1`. "The pipeline got worse" and "I could not tell"
are different problems, and a stale baseline is not a quality signal.

## Capping what a run can spend

`rag-ci run --judge` makes one model call per case, so a large golden set has an
open-ended bill. `--max-cost 5.00` stops before starting further paid work once the spend
reported by your adapter reaches the budget.

```bash
rag-ci run --judge --max-cost 5.00
```

**A budgeted run is slower.** Cost is only known after a call returns, so the run
evaluates cases one at a time rather than concurrently — otherwise every in-flight call
could overshoot together. Measured on twenty cases against a 50 ms retriever, that is
about six times slower than the same run without a budget. Set a budget to bound spend,
not as a precaution you leave on.

**A truncated run cannot be used.** It is marked incomplete, which is distinct from
invalid: the pipeline was healthy, the sample just is not finished. The gate refuses it
and `--update-baseline` refuses it, because half a golden set produces a perfectly
plausible-looking score that means nothing next to a full baseline. `rag-ci run` exits `2`.

Adapters that do not report `cost_usd` cannot be budget-limited — there is nothing to
count, so the run completes normally.

## Flaky pipelines

A run where more than 5% of cases error is marked invalid and the gate exits `2` rather
than reporting a verdict it cannot justify. On a long run against a network-backed
retriever, a handful of timeouts can cross that line for reasons that have nothing to do
with retrieval quality.

`rag-ci run --retries 2` retries a failing case before giving up. Every exception is
retried — an adapter can raise anything, and guessing which failures are transient would
be guessing — so a genuinely broken case costs two extra attempts before it is reported.

**Recovery is reported, not hidden.** A run where cases needed more than one attempt says
so, in the console and in the pull request comment:

```
Note: 7 of 150 cases needed more than one attempt. The scores are real, but the
pipeline is flaky.
```

Retries exist so a blip does not discard good work. They are not there to make an unstable
pipeline look stable, which is why the count is on the report rather than in a log.

## Why `min-effect` exists

The gate requires two things at once: statistical significance **and** a drop larger than
`min-effect`.

Either alone is a trap. Significance alone will fail your build over a uniform 0.005 drop —
which is real, measurable, and completely irrelevant. Effect size alone will fail your build
on random noise from a handful of questions. Teams that hit either failure mode disable the
check within a fortnight, which is how most quality gates actually die.

Tighten `min-effect` when your golden set is large and your pipeline is stable. Loosen it
when a handful of noisy questions keep tripping the build.

## Why the comparison is paired

Per-case scores vary far more between questions than between two versions of a pipeline.
Comparing two independent means buries a real shift under that between-case variance, so
rag-ci compares the same questions scored by both runs and bootstraps the per-case
differences.

The practical consequence: a uniform 0.05 drop across 150 questions is detected at p < 0.01,
while an unpaired comparison of the same data produces confidence intervals that overlap
heavily and looks like nothing happened.

## When the golden set changes

Editing the golden set invalidates the baseline — comparing across two different question
sets measures the questions, not the pipeline. rag-ci detects this by hashing the golden set
into every run record, and exits `2` with an explanation rather than printing a verdict it
cannot justify.

Re-record with `rag-ci gate --update-baseline` and commit the new baseline in the same pull
request as the golden-set change.
