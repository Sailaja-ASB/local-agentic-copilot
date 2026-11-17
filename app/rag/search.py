"""Retrieval for the Copilot: dense, BM25, hybrid (RRF) or hybrid + cross-encoder reranking.

Default is hybrid: on the 47-question benchmark (eval/BENCHMARK.md) dense and BM25 each reached
91.5% Top-1 while hybrid reached 100%; the reranker added ~50 ms/query with no further gain,
so it is available but off by default.

Settings (environment variables):
    RETRIEVAL_MODE        dense | bm25 | hybrid | hybrid_rerank   (default: hybrid)
    RETRIEVAL_CANDIDATES  chunks each retriever passes to fusion  (default: 10)
    CHROMA_PATH           vector database folder                  (default: data/chroma_documents)
    CHROMA_COLLECTION     collection name                         (default: knowledge_base)
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

import chromadb  # Opens our local Chroma vector database.
from rank_bm25 import BM25Okapi  # Lexical (keyword) scoring.
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.rag.fusion import reciprocal_rank_fusion, tokenize

DB_PATH = os.getenv("CHROMA_PATH", "data/chroma_documents")  # Built by indexer.py.
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "knowledge_base")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Same embedding model used during indexing.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "10"))
MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


# Models and the database are loaded on first use, not at import time,
# so importing this module (for example in tests) stays fast.
@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


@lru_cache(maxsize=1)
def _collection():
    return chromadb.PersistentClient(path=DB_PATH).get_collection(name=COLLECTION_NAME)


_lexical_cache: dict = {}


def _lexical_index() -> dict:
    """All chunks from Chroma plus a BM25 index over them, rebuilt only when the chunk count changes."""
    collection = _collection()
    count = collection.count()
    if _lexical_cache.get("count") != count:
        data = collection.get(include=["documents", "metadatas"])
        _lexical_cache.clear()
        _lexical_cache.update(
            count=count,
            chunks={cid: (doc, meta) for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])},
            ids=list(data["ids"]),
            bm25=BM25Okapi([tokenize(doc) for doc in data["documents"]]),
        )
    return _lexical_cache


def _dense(question: str, n: int) -> tuple[list[str], dict[str, float]]:
    embedding = _embedder().encode(question).tolist()
    result = _collection().query(query_embeddings=[embedding], n_results=n)
    ids = result["ids"][0]
    return ids, dict(zip(ids, result["distances"][0]))


def _bm25(question: str, index: dict, n: int) -> list[str]:
    scores = index["bm25"].get_scores(tokenize(question))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [index["ids"][i] for i in order[:n]]


def search_with_timings(question: str, top_k: int = 3, mode: str | None = None) -> tuple[list[dict], dict]:
    """Retrieve the top_k chunks and report how long each stage took (milliseconds)."""
    mode = mode or RETRIEVAL_MODE
    if mode not in MODES:
        raise ValueError(f"Unknown RETRIEVAL_MODE '{mode}'. Use one of {MODES}.")

    timings: dict[str, float] = {}
    index = _lexical_index()
    n = min(max(CANDIDATES, top_k), index["count"])
    if n == 0:
        return [], timings

    dense_ids, distances, bm25_ids = [], {}, []
    if mode != "bm25":
        t = time.perf_counter()
        dense_ids, distances = _dense(question, n)
        timings["dense_ms"] = (time.perf_counter() - t) * 1000
    if mode != "dense":
        t = time.perf_counter()
        bm25_ids = _bm25(question, index, n)
        timings["bm25_ms"] = (time.perf_counter() - t) * 1000

    if mode == "dense":
        ranked = [(cid, -distances[cid]) for cid in dense_ids]
    elif mode == "bm25":
        ranked = [(cid, 1.0 / (rank + 1)) for rank, cid in enumerate(bm25_ids)]
    else:
        ranked = reciprocal_rank_fusion([dense_ids, bm25_ids])

    if mode == "hybrid_rerank":
        t = time.perf_counter()
        candidates = [cid for cid, _ in ranked[:n]]
        scores = _reranker().predict([(question, index["chunks"][cid][0]) for cid in candidates])
        ranked = sorted(zip(candidates, (float(s) for s in scores)), key=lambda pair: pair[1], reverse=True)
        timings["rerank_ms"] = (time.perf_counter() - t) * 1000

    timings["total_ms"] = sum(timings.values())

    dense_rank = {cid: r for r, cid in enumerate(dense_ids, start=1)}
    bm25_rank = {cid: r for r, cid in enumerate(bm25_ids, start=1)}
    matches = []
    for cid, score in ranked[:top_k]:
        text, metadata = index["chunks"][cid]
        page = metadata.get("page", -1)  # PDF page number; TXT/Markdown uses -1.
        matches.append({
            "text": text,
            "source": metadata["source"],
            "page": None if page == -1 else page,
            "chunk_index": metadata["chunk_index"],
            "distance": distances.get(cid),  # None if dense search did not return this chunk.
            "score": score,
            "mode": mode,
            "dense_rank": dense_rank.get(cid),
            "bm25_rank": bm25_rank.get(cid),
        })
    return matches, timings


def search_documents(question: str, top_k: int = 3, mode: str | None = None) -> list[dict]:
    """Retrieve the most relevant document chunks for a question (same interface as before)."""
    return search_with_timings(question, top_k=top_k, mode=mode)[0]


if __name__ == "__main__":
    question = "What does page-aware ingestion preserve?"
    matches, timings = search_with_timings(question, top_k=3)

    print(f"\nQUESTION: {question}")
    print(f"MODE: {RETRIEVAL_MODE} | " + " | ".join(f"{k} {v:.1f}" for k, v in timings.items()))
    for rank, match in enumerate(matches, start=1):
        print(f"\nRank {rank}  (dense rank {match['dense_rank']}, BM25 rank {match['bm25_rank']})")
        print(f"Source: {match['source']}")
        print(f"Page: {match['page']}")
        print(f"Text: {match['text'][:300]}")
