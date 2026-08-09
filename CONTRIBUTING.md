# Contributing

Thanks for looking. This is a young project and the shape of things can still change —
if you are planning something substantial, open an issue first so we can agree on the
approach before you write code.

## Getting set up

```bash
git clone https://github.com/Nokimalos/rag-ci
cd rag-ci
uv sync --dev
uv run pytest
```

That is the whole setup. There is no network access and no API key required: the test
suite runs against an in-memory reference adapter over a toy corpus.

## Before opening a pull request

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

CI runs the same three on Python 3.12, 3.13 and 3.14. `main` is protected, so everything
goes through a pull request.

## How we work

**Tests first.** Write the failing test, watch it fail, then make it pass. This is not
ceremony — a test you never saw fail has not been shown to test anything.

**Metrics are verified against hand-computed values.** Look at `tests/test_metrics.py`:
every case is small enough to work out on paper. A wrong metric invalidates the entire
tool, so "it looks about right" is not a standard we can use here.

**Statistics are verified against known distributions.** `tests/test_stats.py` checks that
the bootstrap covers the true mean at roughly its nominal rate and that intervals narrow as
the sample grows. `stats.py` deliberately knows nothing about retrieval so that it can be
tested this way.

**Comment the why, not the what.** If a line needs explaining, explain the reasoning behind
it. The code already says what it does.

**Conventional commits**: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `ci:`.

## Read this first

[`docs/design.md`](docs/design.md) explains why the architecture is what it is. Several
decisions look arbitrary and are not:

- Ground truth is anchored to **document passages**, never to chunk identifiers. Anchor it
  to chunks and the golden set dies the moment someone changes `chunk_size`, which would
  make configuration sweeps impossible.
- The adapter contract captures a **trajectory** — a list of retrieval steps — rather than
  a single query and its results. Multi-hop and agentic pipelines are first-class.
- The gate blocks only when a regression is **both** statistically significant **and**
  larger than `min-effect`. Either condition alone produces a check that teams disable
  within a fortnight.

If you want to change one of these, that is a conversation worth having — just start it in
an issue rather than in a pull request.

## Adding a metric

1. Write it in `src/ragci/metrics.py` with the signature
   `(trace, case, k, threshold) -> float`.
2. Register it in `TIER1_METRICS`; `parse_metric_name` picks it up automatically.
3. Test it in `tests/test_metrics.py` against values you computed by hand, including the
   empty-trace case.

Metrics must be deterministic and free to compute. Anything needing an LLM call belongs in
tier 2, which is a separate concern.

## Writing an adapter for your own stack

Run `uvx rag-ci init` and fill in `retrieve()`. Report `char_start` and `char_end` on your
chunks whenever you can — with exact offsets rag-ci matches chunks to passages precisely,
and without them it falls back to token overlap and flags the run as degraded.

If your stack makes the contract awkward, that is useful information. Open an issue with
the details.

## Releasing

A release is a pull request that bumps the version in `pyproject.toml`; merging it
publishes to PyPI automatically. See [`docs/releasing.md`](docs/releasing.md).

## Reporting a bug

Include the rag-ci version (`uvx rag-ci --version`), your Python version, and the smallest
adapter that reproduces the problem. The reference adapter in `examples/reference/` is a
good starting point to fork.
