# Building a golden set

A golden set is a list of questions, each anchored to the passage that answers it. It is
what every rag-ci measurement compares against, and the one artefact you genuinely have to
own — so rag-ci helps you build it rather than generating it behind your back.

```bash
uvx "rag-ci[generate]" golden gen --corpus ./docs --sample 20 --max-cases 50
uvx rag-ci golden review
git add golden.jsonl && git commit -m "chore: add golden set"
```

## Corpus formats

**A directory** of `.txt` and `.md` files, walked recursively. The document id is the path
relative to the root (always POSIX-style, so the same corpus hashes identically on every
machine), and the parent folder becomes the `source` metadata used for stratification.

**A JSONL file**, one document per line — the escape hatch for anything else. Export from
your CMS, database, or vector store into:

```json
{"doc_id": "handbook/leave", "text": "Annual leave is 25 days...", "metadata": {"source": "handbook", "lang": "en"}}
```

Any metadata key can then be used with `--stratify-by`.

## Why ground truth is anchored to passages

A golden case does not say "the answer is in chunk 47". It says "the answer is in
`handbook/leave.md`, characters 1200–1480". Chunk numbering changes the moment you change
`chunk_size`; document offsets do not. This is what lets you compare two chunking
strategies against the same golden set at all — see [the design document](design.md).

## Sampling: why stratified

`--sample 20` does not take 20 documents at random. It groups by `--stratify-by`
(`source` by default) and guarantees **every group is represented** before distributing
what is left in proportion to group size.

The failure it prevents: a corpus of 5,000 support tickets and 12 policy documents. Uniform
sampling picks zero policy documents almost every time, and you ship a golden set that never
tests the content people actually search for. Sampling is seeded — same corpus, same
`--seed`, same sample.

The command prints its coverage so you can see what it did:

```
Corpus: 5012 documents across 2 strata; sampled 20.
```

### It does not load your corpus

Deciding what to sample needs a `doc_id` and a stratum key per document — never the text.
So the corpus is read twice: the first pass counts and stratifies while holding only those
identifiers, the second materialises just the documents that were selected. Measured on
20,000 documents of 2 KB each, peak memory is 1.3 MB against 15.4 MB for the naive path,
and the gap widens as documents get larger.

Calling `sample_with_report` from your own code gets the same treatment if you hand it a
callable — `sample_with_report(lambda: load_corpus(path), n=20)`. Passing an iterable
directly still works and still materialises everything, because a bare generator cannot be
walked a second time.

## Generation costs one model call per passage

A sampled document usually yields several passages, and each passage is one API call. Start
small:

```bash
uvx "rag-ci[generate]" golden gen --corpus ./docs --sample 20 --max-cases 50
```

`--sample` bounds how many documents are read; `--max-cases` bounds how many questions are
generated. Run without `--max-cases` only once you have seen what the output looks like.

The default model is `claude-opus-5`; `--model` accepts any Claude model id. Generation
needs `ANTHROPIC_API_KEY` and the optional extra — plain `rag-ci` deliberately installs no
model dependency, so `run` and `gate` work in environments that have no API access at all.

## Review: least confident first

`golden gen` writes to `golden.candidates.jsonl`, not to your golden set. Nothing enters
`golden.jsonl` until you approve it.

```bash
uvx rag-ci golden review
```

Each candidate shows its question, its source passage, and the generator's own confidence.
You can **a**ccept, **e**dit the question then accept, **r**eject, **s**kip, or **q**uit.

Candidates are ordered by **lowest confidence first**. The generator is asked to score its
own output honestly, and the cases most likely to be wrong are therefore the ones you see
while your attention is fresh. If you stop halfway, the tail you did not reach is the part
the generator was most sure about.

Three properties matter in practice:

- **Accepted work is flushed immediately.** Closing the terminal never loses a review.
- **Review resumes.** Re-running picks up at the first undecided candidate.
- **Rejections are remembered.** A rejected candidate does not come back next time — only
  `skip` leaves a case pending on purpose.

## Growing the set over time

A golden set of 30 reviewed cases is already useful; 200 gives you tight confidence
intervals. Generate in batches, review what you have patience for, and commit as you go.

Remember that **editing the golden set invalidates the baseline** — rag-ci stores its hash
in every run record and refuses to compare across two different question sets. After adding
cases, re-record with `rag-ci gate --update-baseline` in the same pull request.
