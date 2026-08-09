import re
import tomllib
from pathlib import Path

import pytest

import ragci

# Every way a release version is written down outside pyproject.toml.
PINNED = re.compile(
    r"rag-ci@v(\d+\.\d+\.\d+)"  # uses: Nokimalos/rag-ci@vX.Y.Z
    r'|rag-ci==(\d+\.\d+\.\d+)"'  # uvx "rag-ci==X.Y.Z"
    r'|RAGCI_VERSION: "(\d+\.\d+\.\d+)"'  # the action and the GitLab job
    # The bug report template also has a Python-version placeholder, so anchor on the
    # `rag-ci --version` description that precedes ours.
    r'|rag-ci --version`\n\s+placeholder: "(\d+\.\d+\.\d+)"'
)

VERSIONED_FILES = [
    "README.md",
    "action.yml",
    "docs/github-action.md",
    "docs/gitlab-ci.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
]


def test_package_exposes_a_version():
    assert isinstance(ragci.__version__, str)
    assert ragci.__version__.count(".") == 2


def test_the_reported_version_matches_pyproject():
    # A release that bumps pyproject but not the reported version ships a package
    # that lies about itself in every bug report and every `--version` call.
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert ragci.__version__ == declared


def test_the_action_pins_the_version_it_ships_with():
    # `uses: rag-ci@vX.Y.Z` must run CLI X.Y.Z, not whatever is newest on PyPI.
    root = Path(__file__).resolve().parents[1]
    action = (root / "action.yml").read_text(encoding="utf-8")
    assert f'RAGCI_VERSION: "{ragci.__version__}"' in action


@pytest.mark.parametrize("relative", VERSIONED_FILES)
def test_every_pinned_version_matches_the_release(relative):
    # A bump that misses one of these leaves a copy-pasteable snippet pointing at an
    # older release, which is worse than no snippet: it looks current.
    path = Path(__file__).resolve().parents[1] / relative
    found = {
        v
        for match in PINNED.finditer(path.read_text(encoding="utf-8"))
        for v in match.groups()
        if v
    }
    assert found, f"{relative} is listed as version-bearing but pins nothing"
    assert found == {ragci.__version__}, f"{relative} pins {sorted(found)}"
