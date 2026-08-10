# Using rag-ci in GitLab CI

There is no GitLab component to install. `rag-ci run` and `rag-ci gate` are a plain CLI that
reads files and returns an exit code, so the whole integration is one job.

Add this to `.gitlab-ci.yml`:

```yaml
rag-ci:
  image: ghcr.io/astral-sh/uv:python3.12-bookworm-slim
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  variables:
    RAGCI_VERSION: "0.5.0"
  # 1 = real regression, blocks the pipeline. 2 = comparison not trustworthy, warns.
  allow_failure:
    exit_codes: 2
  script:
    - uvx "rag-ci==${RAGCI_VERSION}" run
        --adapter ragci_adapter.py
        --golden tests/golden.jsonl
        --out .ragci/run.json
    - uvx "rag-ci==${RAGCI_VERSION}" gate
        --run .ragci/run.json
        --baseline .ragci/baseline.json
        --min-effect 0.02
        --markdown .ragci/report.md
  artifacts:
    when: always
    paths:
      - .ragci/run.json
      - .ragci/report.md
```

`when: always` on the artifacts matters: the report is most useful on the run that just
failed, which is exactly when GitLab would otherwise discard it.

Pin `RAGCI_VERSION` rather than tracking the newest release. A gate whose implementation
changes under you is a gate whose verdicts you cannot compare over time.

## Exit codes map better here than on GitHub

`allow_failure: exit_codes: 2` is the reason this integration reads more cleanly on GitLab.
The gate distinguishes two outcomes that deserve different treatment:

| Code | Meaning | GitLab result |
|---|---|---|
| `0` | Pass — no regression, or no baseline yet | ✅ passed |
| `1` | Regression, significant **and** larger than `min-effect` | ❌ failed |
| `2` | Could not compare — invalid run, or stale baseline | ⚠️ passed with warning |

"The pipeline got worse" and "I could not tell" are different problems. GitHub Actions has
no equivalent of `exit_codes`, so both surface as a red job there and the distinction
survives only in the report. GitLab keeps it in the pipeline itself.

## Posting the report on the merge request

This is the one place GitLab needs more setup than GitHub, where `github.token` covers it
with no configuration. `CI_JOB_TOKEN` cannot write merge request notes, so you need a
[project access token](https://docs.gitlab.com/user/project/settings/project_access_tokens/)
with the `api` scope, exposed as a masked CI variable named `RAGCI_MR_TOKEN`.

Add to the job:

```yaml
  after_script:
    - |
      { test -f .ragci/report.md && test -n "$RAGCI_MR_TOKEN" && curl -sS --request POST \
          --header "PRIVATE-TOKEN: $RAGCI_MR_TOKEN" \
          --form "body=<.ragci/report.md" \
          "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/notes"; } || true
```

`after_script` runs whether or not the gate passed, which is the behaviour you want — a
blocked merge request is the one that most needs its report.

The `|| true` makes a missing token degrade quietly instead of turning a clean verdict into
a red pipeline for the wrong reason. Without the token the gate still decides, and the
report is still downloadable from the job artifacts.

## Speeding it up

`uvx` resolves and downloads rag-ci on every run. GitLab only caches paths inside the
project directory, so point uv's cache there:

```yaml
  variables:
    RAGCI_VERSION: "0.5.0"
    UV_CACHE_DIR: .uv-cache
  cache:
    key:
      files:
        - .gitlab-ci.yml
    paths:
      - .uv-cache
```

Keying on `.gitlab-ci.yml` invalidates the cache when you bump `RAGCI_VERSION`, which is the
only time a stale cache would matter.

## Everything else is the same

Recording the first baseline, choosing `min-effect`, why the comparison is paired, and what
happens when the golden set changes are not CI-platform concerns. Those sections of
[docs/github-action.md](github-action.md) apply here word for word.

The short version: the gate has nothing to compare against until a baseline exists, so run
it once locally and commit the result.

```bash
uv run rag-ci run --adapter ragci_adapter.py --golden tests/golden.jsonl
uv run rag-ci gate --update-baseline
git add .ragci/baseline.json && git commit -m "chore: record rag-ci baseline"
```
