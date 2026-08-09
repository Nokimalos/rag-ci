"""Reading a corpus without ever holding it in memory."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

TEXT_SUFFIXES = {".txt", ".md"}


class CorpusError(Exception):
    """The corpus could not be read. The message says what to do about it."""


class Document(BaseModel):
    doc_id: str = Field(min_length=1)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def _load_directory(root: Path) -> Iterator[Document]:
    found = False
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        found = True
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        relative = path.relative_to(root)
        yield Document(
            # POSIX-style so a corpus hashes identically on every platform.
            doc_id=relative.as_posix(),
            text=text,
            metadata={"source": relative.parent.as_posix()},
        )
    if not found:
        raise CorpusError(f"{root} contains no .txt or .md files")


def _load_jsonl(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            document = Document.model_validate_json(line)
            if document.text.strip():
                yield document


def load_corpus(path: Path) -> Iterator[Document]:
    """Stream a corpus from a directory of text files or a JSONL export."""
    path = Path(path)
    if not path.exists():
        raise CorpusError(f"{path} does not exist")
    if path.is_dir():
        yield from _load_directory(path)
    else:
        yield from _load_jsonl(path)
