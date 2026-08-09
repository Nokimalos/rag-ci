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
      - uses: Nokimalos/rag-ci@v0.4.2
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
