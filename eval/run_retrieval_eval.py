"""Retrieval benchmark: dense vs BM25 vs hybrid (RRF) vs hybrid + cross-encoder reranker.

Same 47 questions, same chunks, four methods. A question counts as correct when its
expected source file is ranked first (Top-1) or appears in the first 3 distinct
sources (Hit@3). MRR = average of 1 / rank of the expected source.

Run from the repo root:
    python eval/run_retrieval_eval.py
    python eval/run_retrieval_eval.py --candidates 20 --show-failures

Outputs:
    eval/results/retrieval_<timestamp>.json   every question, every method, full ranking
    eval/results/RESULTS.md                   summary table (latest run)
"""
import argparse
import json
import platform
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.rag.ingestion import load_documents  # noqa: E402  same chunking as the app (500 chars, 100 overlap)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
METHODS = ["dense", "bm25", "hybrid_rrf", "hybrid_rerank"]


def tokenize(text: str) -> list[str]:
    # Keeps identifiers like max_inflight, ollama_num_parallel and all-minilm-l6-v2 as tokens.
    return re.findall(r"[a-z0-9]+(?:[_\-.][a-z0-9]+)*", text.lower())


def to_sources(chunk_ids: list[int], chunks: list[dict]) -> list[str]:
    """Chunk ranking -> source ranking (first occurrence of each file wins)."""
    seen, sources = set(), []
    for i in chunk_ids:
        src = chunks[i]["source"]
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def rrf(*rankings: list[int], k: int = RRF_K) -> list[int]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs/benchmark")
    ap.add_argument("--questions", default="eval/questions_47.json")
    ap.add_argument("--candidates", type=int, default=10, help="chunks passed from each retriever to fusion/reranking")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--show-failures", action="store_true")
    args = ap.parse_args()

    chunks = load_documents(ROOT / args.docs)
    questions = json.loads((ROOT / args.questions).read_text(encoding="utf-8"))
    n_sources = len({c["source"] for c in chunks})
    print(f"Corpus: {n_sources} files, {len(chunks)} chunks | Questions: {len(questions)}")

    texts = [c["text"] for c in chunks]
    t0 = time.perf_counter()
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    print(f"Models loaded in {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    doc_emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    index_s = time.perf_counter() - t0

    n = args.candidates
    per_question, timings = [], defaultdict(list)

    for q in questions:
        text, expected = q["question"], q["expected_source"]
        rankings = {}

        t = time.perf_counter()
        q_emb = embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        dense = list(np.argsort(-(doc_emb @ q_emb)))
        timings["dense"].append(time.perf_counter() - t)
        rankings["dense"] = dense

        t = time.perf_counter()
        lexical = list(np.argsort(-bm25.get_scores(tokenize(text))))
        timings["bm25"].append(time.perf_counter() - t)
        rankings["bm25"] = lexical

        t = time.perf_counter()
        fused = rrf(dense[:n], lexical[:n])
        timings["hybrid_rrf"].append(timings["dense"][-1] + timings["bm25"][-1] + time.perf_counter() - t)
        rankings["hybrid_rrf"] = fused

        t = time.perf_counter()
        cands = fused[:n]
        scores = reranker.predict([(text, texts[i]) for i in cands], show_progress_bar=False)
        reranked = [cands[i] for i in np.argsort(-np.asarray(scores))]
        timings["hybrid_rerank"].append(timings["hybrid_rrf"][-1] + time.perf_counter() - t)
        rankings["hybrid_rerank"] = reranked

        row = {**q, "results": {}}
        for m in METHODS:
            srcs = to_sources([int(i) for i in rankings[m]], chunks)
            rank = srcs.index(expected) + 1 if expected in srcs else None
            row["results"][m] = {"rank": rank, "top3": srcs[:3]}
        per_question.append(row)

    # ---------- Summary ----------
    total = len(per_question)
    summary = {}
    for m in METHODS:
        ranks = [r["results"][m]["rank"] for r in per_question]
        summary[m] = {
            "top1": sum(1 for r in ranks if r == 1),
            "hit3": sum(1 for r in ranks if r and r <= 3),
            "mrr": sum(1 / r for r in ranks if r) / total,
            "ms_per_query": 1000 * float(np.mean(timings[m])),
        }
    types = sorted({q["type"] for q in per_question})
    by_type = {m: {t: [0, 0] for t in types} for m in METHODS}
    for r in per_question:
        for m in METHODS:
            by_type[m][r["type"]][1] += 1
            by_type[m][r["type"]][0] += int(r["results"][m]["rank"] == 1)

    lines = [
        "# Retrieval benchmark results",
        "",
        f"Run: {datetime.now():%Y-%m-%d %H:%M} | Machine: {platform.machine()} {platform.system()} | "
        f"Python {platform.python_version()}",
        f"Corpus: {n_sources} files, {len(chunks)} chunks (500 chars, 100 overlap) | Questions: {total} | "
        f"Candidates per retriever: {n} | Index build: {index_s:.2f} s",
        f"Embeddings: `{EMBEDDING_MODEL}` | Reranker: `{RERANKER_MODEL}` | RRF k = {RRF_K}",
        "",
        "| Method | Top-1 | Hit@3 | MRR | ms / query |",
        "|---|---|---|---|---|",
    ]
    for m in METHODS:
        s = summary[m]
        lines.append(f"| {m} | {s['top1']}/{total} ({100 * s['top1'] / total:.1f}%) | "
                     f"{s['hit3']}/{total} ({100 * s['hit3'] / total:.1f}%) | {s['mrr']:.3f} | "
                     f"{s['ms_per_query']:.1f} |")
    lines += ["", "### Top-1 by question type", "",
              "| Method | " + " | ".join(f"{t}" for t in types) + " |",
              "|---|" + "---|" * len(types)]
    for m in METHODS:
        lines.append(f"| {m} | " + " | ".join(f"{by_type[m][t][0]}/{by_type[m][t][1]}" for t in types) + " |")

    fails = [r for r in per_question if r["results"]["hybrid_rerank"]["rank"] != 1]
    lines += ["", f"### Questions hybrid + rerank did not rank first ({len(fails)})", ""]
    for r in fails:
        res = r["results"]["hybrid_rerank"]
        lines.append(f"- {r['id']} [{r['type']}] {r['question']}  \n  expected `{r['expected_source']}`, "
                     f"rank {res['rank']}, got {res['top3']}")

    report = "\n".join(lines)
    print("\n" + report)

    if args.show_failures:
        print("\n### All failures by method")
        for m in METHODS:
            for r in per_question:
                res = r["results"][m]
                if res["rank"] != 1:
                    print(f"{m:14} {r['id']} rank={res['rank']} expected={r['expected_source']} got={res['top3']}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"retrieval_{stamp}.json").write_text(json.dumps(
        {"summary": summary, "by_type": by_type, "config": {
            "embedding_model": EMBEDDING_MODEL, "reranker_model": RERANKER_MODEL, "rrf_k": RRF_K,
            "candidates": n, "chunks": len(chunks), "files": n_sources},
         "questions": per_question}, indent=2), encoding="utf-8")
    (out / "RESULTS.md").write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {out / 'RESULTS.md'} and retrieval_{stamp}.json")


if __name__ == "__main__":
    main()
