from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: pyproject.toml. Hardcoding it here means a release can
    # ship a package that misreports its own version in every bug report.
    __version__ = version("rag-ci")
except PackageNotFoundError:  # running from a source tree with nothing installed
    __version__ = "0.0.0+unknown"
