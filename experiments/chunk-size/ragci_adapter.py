"""rag-ci against a real vector pipeline: LangChain splitter + Chroma + ONNX embeddings.

The one thing that makes this work is `add_start_index=True` on the splitter. Without it
LangChain reports no offsets, rag-ci falls back to token overlap, and the run is flagged
as degraded — the scores still come out, they just measure less precisely.
"""

from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragci.contract import Chunk, ParamSpec, RetrievalTrace, Step, adapter

CORPUS = Path(__file__).parent / "corpus"


@adapter(
    index_time_params=[ParamSpec(name="chunk_size", values=[256, 512, 1024])],
    query_time_params=[ParamSpec(name="top_k", values=[5, 10])],
    primary_metric="recall@10",
)
class LangChainChroma:
    def build_index(self, corpus, config: dict):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("chunk_size", 400),
            chunk_overlap=40,
            add_start_index=True,  # without this, every offset below is None
        )
        # EphemeralClient() hands back the same in-process client every time, so a second
        # build_index for the same parameters finds its collection already there. A sweep
        # rebuilds an index once per rung, so build_index has to be idempotent.
        client = chromadb.EphemeralClient()
        name = f"ragci-{config.get('chunk_size', 400)}"
        if name in [c.name for c in client.list_collections()]:
            client.delete_collection(name)
        collection = client.create_collection(name)

        ids, texts, metas = [], [], []
        for path in sorted(CORPUS.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            for n, piece in enumerate(splitter.create_documents([text])):
                start = piece.metadata["start_index"]
                ids.append(f"{path.name}:{n}")
                texts.append(piece.page_content)
                metas.append(
                    {
                        "doc_id": path.name,
                        "char_start": start,
                        "char_end": start + len(piece.page_content),
                    }
                )
        # Chroma caps a single add() at 5461 records, and a modest corpus blows past that
        # — 1.6 MB at chunk_size=256 is over 8000 chunks. Batch, or it dies at scale only.
        BATCH = 5000
        for i in range(0, len(ids), BATCH):
            collection.add(
                ids=ids[i : i + BATCH],
                documents=texts[i : i + BATCH],
                metadatas=metas[i : i + BATCH],
            )
        return collection

    def retrieve(self, query: str, index, config: dict) -> RetrievalTrace:
        found = index.query(query_texts=[query], n_results=config.get("top_k", 3))
        chunks = [
            Chunk(
                text=text,
                doc_id=meta["doc_id"],
                char_start=meta["char_start"],
                char_end=meta["char_end"],
                score=1.0 - distance,
            )
            for text, meta, distance in zip(
                found["documents"][0], found["metadatas"][0], found["distances"][0], strict=True
            )
        ]
        return RetrievalTrace(steps=[Step(query=query, chunks=chunks)], latency_ms=1.0)
