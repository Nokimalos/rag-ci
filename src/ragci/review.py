"""Promoting generated candidates into a golden set a team can own."""

import json
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

from ragci.golden import GoldenCase, load_golden, save_golden

ReviewDecision = Literal["accept", "reject", "skip", "quit"]
Prompt = Callable[[GoldenCase], tuple[ReviewDecision, str | None]]


class ReviewStats(BaseModel):
    reviewed: int = 0
    accepted: int = 0
    rejected: int = 0


def review_order(cases: Iterable[GoldenCase]) -> list[GoldenCase]:
    """Least confident first — the cases most likely to be wrong get fresh attention."""
    return sorted(cases, key=lambda case: float(case.strata.get("confidence", -1.0)))


class ReviewSession:
    """Candidates on one side, the accepted golden set on the other."""

    def __init__(self, candidates: Path, golden: Path):
        self.candidates = Path(candidates)
        self.golden = Path(golden)
        self._accepted: list[GoldenCase] = (
            list(load_golden(self.golden)) if self.golden.exists() else []
        )
        self._decisions_path = self.candidates.with_suffix(
            self.candidates.suffix + ".decisions.json"
        )
        self._decisions: dict[str, str] = (
            json.loads(self._decisions_path.read_text(encoding="utf-8"))
            if self._decisions_path.exists()
            else {}
        )

    def pending(self) -> list[GoldenCase]:
        done = {case.id for case in self._accepted} | set(self._decisions)
        return review_order(c for c in load_golden(self.candidates) if c.id not in done)

    def record(
        self, case: GoldenCase, decision: ReviewDecision, edited_question: str | None = None
    ) -> None:
        if decision == "skip":
            return  # deliberately undecided: comes back next run
        # Record rejections too, or the reviewer re-judges the same bad case forever.
        self._decisions[case.id] = decision
        self._decisions_path.parent.mkdir(parents=True, exist_ok=True)
        self._decisions_path.write_text(json.dumps(self._decisions, indent=2), encoding="utf-8")
        if decision != "accept":
            return
        if edited_question:
            case.question = edited_question
        case.provenance = "synthetic-reviewed"
        case.reviewed_at = date.today().isoformat()
        self._accepted.append(case)
        # Flush on every acceptance: a closed terminal must never lose reviewed work.
        self.golden.parent.mkdir(parents=True, exist_ok=True)
        save_golden(self.golden, self._accepted)


def _show(case: GoldenCase, console: Console) -> None:
    passage = case.required_passages[0]
    confidence = case.strata.get("confidence")
    console.print(
        Panel(
            f"[bold]{case.question}[/]\n\n"
            f"[dim]{passage.doc_id} [{passage.char_start}:{passage.char_end}][/]\n\n"
            f"{passage.text}",
            title=f"confidence {confidence:.2f}" if confidence is not None else "unscored",
        )
    )


def run_review(session: ReviewSession, prompt: Prompt, console: Console | None) -> ReviewStats:
    stats = ReviewStats()
    for case in session.pending():
        if console is not None:
            _show(case, console)
        decision, edited = prompt(case)
        if decision == "quit":
            break
        session.record(case, decision, edited)
        if decision == "skip":
            continue
        stats.reviewed += 1
        if decision == "accept":
            stats.accepted += 1
        else:
            stats.rejected += 1
    return stats
