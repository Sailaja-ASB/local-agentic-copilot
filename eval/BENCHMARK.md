# Retrieval Benchmark (47 questions)

A fixed benchmark to compare retrieval methods fairly. Everything needed to rerun it is in this repo.

## Dataset
- **Corpus:** `data/docs/benchmark/`: 18 short technical documents with deliberately overlapping topics
  (vector search vs keyword search vs hybrid, load shedding vs rate limiting, probes vs autoscaling, …).
- **Questions:** `eval/questions_47.json`: 47 questions, each with one expected source file and a type:

| Type | Count | What it tests |
|---|---|---|
| direct | 12 | wording close to the source |
| paraphrase | 14 | same meaning, different words |
| keyword | 12 | exact identifiers (`MAX_INFLIGHT`, `OLLAMA_NUM_PARALLEL`, HTTP 429, RRF k) |
| confusable | 9 | topics that appear in more than one document |

Questions and expected sources were written **before** any method was run, and they are not tuned to the results.

## Methods (same chunks: 500 characters, 100 overlap)
1. **dense**: `all-MiniLM-L6-v2` embeddings, cosine similarity
2. **bm25**: BM25Okapi over tokens that keep identifiers like `max_inflight` intact
3. **hybrid_rrf**: Reciprocal Rank Fusion (k = 60) of the dense and BM25 top 10
4. **hybrid_rerank**: hybrid top 10 re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2` (question + chunk text only)

## Metrics
- **Top-1:** expected source ranked first
- **Hit@3:** expected source among the first 3 distinct sources
- **MRR:** mean of 1 / rank of the expected source
- **ms / query:** average retrieval time on the machine that ran it

## Run
```bash
python eval/run_retrieval_eval.py --show-failures
```
Results: `eval/results/RESULTS.md` (latest summary) and `eval/results/retrieval_<timestamp>.json` (every question, every method).

## Limitations
- 47 questions: one question = 2.1 percentage points, so small differences are noise.
- A single relevant source per question; real queries can have several.
- Short documents (1–3 chunks each); longer documents make chunking choices matter more.
