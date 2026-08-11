"""Build the corpus and golden set for the chunk-size experiment, from SQuAD v1.1.

SQuAD answers are literal spans of the source, and the dataset ships the offset of each
one — so the golden set is exact by construction rather than by our own matching. One
document per Wikipedia article, not per paragraph, so retrieval has something to do.

    uv run --with requests python prepare.py
"""

import json
import pathlib
import random
import urllib.request

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SAMPLE = 300
SEED = 0
SEPARATOR = "\n\n"

here = pathlib.Path(__file__).parent
corpus = here / "corpus"
corpus.mkdir(exist_ok=True)

raw = here / "dev-v1.1.json"
if not raw.exists():
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, raw)  # noqa: S310 - fixed, well-known URL

articles = json.loads(raw.read_text(encoding="utf-8"))["data"]
pairs, golden = [], []

for article in articles:
    doc_id = article["title"].replace("/", "_") + ".txt"
    text, placed = "", []
    for paragraph in article["paragraphs"]:
        placed.append((len(text), paragraph))
        text += paragraph["context"] + SEPARATOR
    (corpus / doc_id).write_text(text, encoding="utf-8")

    for base, paragraph in placed:
        for qa in paragraph["qas"]:
            answer = qa["answers"][0]
            start = base + answer["answer_start"]
            # Verify rather than trust: the offset must address the concatenated document.
            if text[start : start + len(answer["text"])] != answer["text"]:
                continue
            pairs.append({"id": qa["id"], "question": qa["question"], "answer": answer["text"]})
            golden.append(
                {
                    "id": qa["id"],
                    "question": qa["question"],
                    "required_passages": [
                        {
                            "doc_id": doc_id,
                            "char_start": start,
                            "char_end": start + len(answer["text"]),
                            "text": answer["text"],
                        }
                    ],
                }
            )

chosen = random.Random(SEED).sample(range(len(pairs)), SAMPLE)
(here / "qa.jsonl").write_text("\n".join(json.dumps(pairs[i]) for i in chosen))
(here / "golden.jsonl").write_text("\n".join(json.dumps(golden[i]) for i in chosen))

sizes = [len(p.read_text(encoding="utf-8")) for p in corpus.glob("*.txt")]
print(f"{len(sizes)} documents, {sum(sizes) // 1000} k characters")
print(f"{len(pairs)} usable pairs, sampled {SAMPLE}")
