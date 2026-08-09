import tomllib
from pathlib import Path

import ragci


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
