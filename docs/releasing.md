# Releasing

A release is a pull request that bumps the version. Merging it publishes.

## The process

1. Bump the version in **`pyproject.toml`** and in the two `RAGCI_VERSION` values in
   **`action.yml`** — `tests/test_version.py` fails if they disagree, so a partial bump
   never reaches `main`.
2. Update the `@vX.Y.Z` references in `README.md` and `docs/github-action.md`, and the
   placeholder in the issue template.
3. Add a `CHANGELOG.md` entry.
4. Open the pull request, get it green, merge it.

That is the whole process. `release.yml` sees `pyproject.toml` change on `main`, and runs
the tests again, builds, publishes to PyPI over OIDC, then tags the commit and creates the
GitHub release.

## Why the bump is the trigger, and not a tag

A workflow that pushes a tag using the default `GITHUB_TOKEN` **does not trigger other
workflows** — GitHub blocks the cascade to prevent loops. A tag-then-release pair would
therefore post the tag and never publish, silently. So the tag is an *output* of the
release, produced after the package is on PyPI, rather than the thing that starts it.

Ordering matters for the same reason: PyPI is the irreversible step and runs first. A tag
that existed without a published package would make the next run think the version was
already released and skip it.

## Idempotence

The workflow asks PyPI *"is this version already published?"* — not git, and not "did the
version change?". Re-running it on an already-released version does nothing, and an
unrelated edit to `pyproject.toml` triggers the workflow, finds the version published, and
stops.

**Why the index and not the tag.** A tag proves someone tagged; only the index proves the
package shipped, and publication is the step that must never happen twice. Version 0.3.0
was tagged by hand during a workflow migration and published nowhere — a tag-based check
would have reported it as released forever. The simple index is queried rather than the
JSON API, which is CDN-cached and lagged by several minutes during the 0.2.0 release.

A hand-pushed tag still triggers the workflow too. A release process must not depend on
the order two pull requests happen to be merged in.

## Trying it without publishing

Run the workflow manually from the Actions tab with `dry_run` left ticked. It reports what
it would release in the run summary and stops before tagging or publishing.

## When to release

**When `main` documents a capability the published package does not have.** That is the
objective threshold: a README that describes an absent command wastes a reader's time
before it tells them anything.

In practice that works out to one release per delivered feature, not one per merge.
Internal changes — docs layout, CI fixes, test refactors — change nothing for someone
installing the package and can wait for the next train.

## The one thing that cannot be undone

**A PyPI version number can never be reused.** A version can be yanked; its number is spent
either way. This is why the workflow re-runs the full suite before building, even though CI
already passed on the same commit, and why the version-consistency tests exist: a package
that misstates its own version corrupts every bug report filed against it.
