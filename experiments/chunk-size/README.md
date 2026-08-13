# Does chunk size actually matter?

Six chunking configurations, 300 questions, one real pipeline. The apparent spread is
**12 points of recall**. After a paired test on held-out cases, none of them is
measurably better than the next.

## What was measured

| | |
|---|---|
| Corpus | 48 Wikipedia articles from SQuAD v1.1, 1.6 MB, one document per article |
| Questions | 300, sampled with a fixed seed; answers are literal spans, so offsets are exact |
| Pipeline | LangChain `RecursiveCharacterTextSplitter` → Chroma → ONNX `all-MiniLM-L6-v2` |
| Grid | `chunk_size` ∈ {256, 512, 1024} × `top_k` ∈ {5, 10} |
| Metric | recall@10, meaning the answer passage appears in the retrieved chunks |

## The ranking, which is the part people usually stop at

On the first rung, 75 questions each:

| configuration | recall@10 |
|---|---|
| chunk_size=1024, top_k=10 | **0.960** |
| chunk_size=512, top_k=10 | 0.920 |
| chunk_size=1024, top_k=5 | 0.880 |
| chunk_size=256, top_k=10 | 0.853 |
| chunk_size=512, top_k=5 | 0.853 |
| chunk_size=256, top_k=5 | 0.840 |

Top to bottom, 12 points. It looks like 1024 wins and 256 is a mistake.

## The part that changes the conclusion

The two survivors were then run on 225 questions, and the winner was tested against the
runner-up on **75 questions the search had never scored them against**:

```
chunk_size=1024, top_k=10   0.920
chunk_size=512,  top_k=10   0.889

No clear winner on 75 held-out cases:
  vs chunk_size=512, top_k=10: +0.040 (95% CI [-0.013, 0.107], p=0.1190)
```

The interval crosses zero. On this corpus, with 300 questions, **you cannot say 1024 beats
512** — and those were the two best of the six.

## What this does and does not show

It does **not** show that chunk size is irrelevant. It shows that a 4-point lead over 75
questions is inside the noise, and that the 12-point spread in the first table is mostly
an artifact of measuring six things on 75 questions each.

The uncomfortable implication: most chunk-size decisions are made on far fewer than 300
questions, with no interval at all. A team that picked 1024 here would not be wrong — they
would be unable to know whether they were right.

To conclude either way you need more questions, not more configurations.

## Reproducing it

```bash
uv run --with requests python prepare.py     # downloads SQuAD (5 MB), builds corpus + golden set
uvx --with chromadb --with langchain-text-splitters \
  rag-ci sweep --metric recall@10 --min-cases 75 --holdout 0.25 --report sweep.html
```

`--min-cases 75` matters. At the default of 10, every configuration ties on the first rung
and the cut is decided by tie-break rather than evidence — rag-ci says so rather than
reporting a winner, which is how this experiment started.

Numbers here come from rag-ci 0.7.1 on an M-series Mac. Retrieval is deterministic given
the same corpus, seed and model, so they should reproduce exactly.
