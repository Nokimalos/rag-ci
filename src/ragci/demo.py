"""A corpus, a golden set, and a regression — computed on your machine, not scripted.

The point of a demo here is narrow and awkward: it has to show a gate catching a real
regression, and this tool's whole argument is that you should not believe a number you
did not measure. Printing a canned transcript would contradict the product.

So everything below is generated deterministically and then actually run. The corpus is
synthetic — nothing to download, nothing to license — but the retrieval, the metrics, the
bootstrap and the verdict are the same code paths `rag-ci run` and `rag-ci gate` use.

The corpus is built to be *confusable* on purpose. Facts reuse each other's vocabulary, so
lexical retrieval ranks several plausible documents and the right one is not always first.
A corpus of unrelated facts would score a perfect 1.000 at every setting and demonstrate
nothing at all.
"""

import random

from ragci.contract import Chunk, ParamSpec, RetrievalTrace, Step, adapter
from ragci.corpus import Document
from ragci.golden import GoldenCase, Passage

PLACES = (
    "Meridian Ashford Calder Thornbury Westmoor Ravenna Holloway Kestrel "
    "Aldermere Bexley Cranwich Duxton"
).split()
THINGS = (
    "archive observatory foundry granary library workshop reservoir seminary "
    "almshouse bellfoundry conservatory dockyard"
).split()
UNITS = "volumes instruments castings bushels manuscripts tools litres charters".split()

# Filler that names *other* places and things. Without it every question is uniquely
# answerable by keyword and recall sits at a useless 1.000 whatever top_k you pick.
FILLER = (
    "The {place} {thing} was extended twice",
    "Records held at the {place} {thing} are incomplete for that period",
    "A survey of the {place} {thing} was carried out the following spring",
    "Local accounts of the {place} {thing} disagree on the date",
)


def _facts(seed: int) -> list[tuple[str, str, str, int]]:
    """Every (place, thing) pair, so vocabulary is shared and retrieval has to work."""
    rng = random.Random(seed)
    pairs = [(place, thing) for place in PLACES for thing in THINGS]
    rng.shuffle(pairs)
    return [
        (place, thing, UNITS[i % len(UNITS)], 1000 + rng.randrange(9000))
        for i, (place, thing) in enumerate(pairs)
    ]


def build_corpus(seed: int = 0) -> tuple[list[Document], list[GoldenCase]]:
    """One document per fact, and one question per fact, anchored to the fact's offsets."""
    rng = random.Random(seed + 1)
    documents: list[Document] = []
    cases: list[GoldenCase] = []

    for index, (place, thing, unit, count) in enumerate(_facts(seed)):
        fact = f"The {place} {thing} holds {count} {unit}."

        def noise(n: int) -> str:
            return " ".join(
                rng.choice(FILLER).format(place=rng.choice(PLACES), thing=rng.choice(THINGS)) + "."
                for _ in range(n)
            )

        # The fact sits after some filler, so char offsets are not trivially zero.
        text = f"{noise(2)} {fact} {noise(1)}"
        start = text.index(fact)

        doc_id = f"records/{place.lower()}-{thing}.txt"
        documents.append(Document(doc_id=doc_id, text=text, metadata={"source": place.lower()}))
        cases.append(
            GoldenCase(
                id=f"q{index:03d}",
                question=f"How many {unit} does the {place} {thing} hold?",
                required_passages=[
                    Passage(
                        doc_id=doc_id,
                        char_start=start,
                        char_end=start + len(fact),
                        text=fact,
                    )
                ],
            )
        )
    return documents, cases


@adapter(
    index_time_params=[ParamSpec(name="chunk_size", values=[120, 240])],
    query_time_params=[ParamSpec(name="top_k", values=[1, 3, 5])],
    primary_metric="recall@3",
)
class DemoRag:
    """Plain lexical retrieval — the point is the measurement, not the retriever."""

    def __init__(self, documents: list[Document]):
        self._documents = documents

    def build_index(self, corpus, config: dict) -> list[Chunk]:
        size = config.get("chunk_size", 240)
        return [
            Chunk(
                text=document.text[offset : offset + size],
                doc_id=document.doc_id,
                char_start=offset,
                char_end=min(offset + size, len(document.text)),
            )
            for document in (corpus or self._documents)
            for offset in range(0, len(document.text), size)
        ]

    def retrieve(self, query: str, index, config: dict) -> RetrievalTrace:
        if index is None:
            index = self.build_index(None, config)
        terms = {word.lower().strip("?") for word in query.split() if len(word) > 3}

        scored = sorted(
            (
                (len(terms & {w.lower().strip(".") for w in chunk.text.split()}), chunk)
                for chunk in index
            ),
            key=lambda pair: -pair[0],
        )
        top = [
            c.model_copy(update={"score": float(s)}) for s, c in scored[: config.get("top_k", 5)]
        ]
        return RetrievalTrace(steps=[Step(query=query, chunks=top)], latency_ms=1.0)


BASELINE = {"chunk_size": 240, "top_k": 5}
CANDIDATE = {"chunk_size": 120, "top_k": 5}
METRIC = "recall@3"


async def run_demo(console) -> int:
    """Build the corpus, measure twice, gate. Returns the exit code the gate would use."""
    from ragci.baseline import decide
    from ragci.report import render_console
    from ragci.runner import run_cases

    documents, cases = build_corpus()
    instance = DemoRag(documents)
    console.print(
        f"Built a synthetic corpus: [bold]{len(documents)}[/] documents, "
        f"[bold]{len(cases)}[/] questions anchored to character offsets.\n"
        "[dim]Nothing downloaded, no API key. Everything below is measured as it runs.[/]\n"
    )

    async def measure(config: dict):
        return await run_cases(
            instance,
            cases,
            config=config,
            metric_names=[METRIC],
            golden_hash="demo",
            index=instance.build_index(None, config),
        )

    console.print(f"[bold]1.[/] Baseline — chunk_size={BASELINE['chunk_size']}")
    baseline = await measure(BASELINE)
    render_console(baseline, console)

    console.print(
        f"\n[bold]2.[/] Someone halves the chunk size to fit more of them in context "
        f"— chunk_size={CANDIDATE['chunk_size']}"
    )
    candidate = await measure(CANDIDATE)
    render_console(candidate, console)

    console.print("\n[bold]3.[/] The gate compares them over the same questions")
    decision = decide(baseline, candidate, metric=METRIC, min_effect=0.02)
    verdict = "[green]Pass[/]" if decision.passed else "[red]REGRESSION — the gate fails[/]"
    console.print(f"{verdict} — {decision.message}")

    console.print(
        "\n[dim]That verdict came from a paired bootstrap over the per-question scores, "
        "not from comparing two averages. Point rag-ci at your own pipeline with "
        "`rag-ci init`.[/]"
    )
    return 0 if decision.passed else 1
