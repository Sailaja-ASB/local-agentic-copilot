# Retrieval benchmark results

Run: 2026-11-16 19:14 | Machine: arm64 Darwin | Python 3.9.6
Corpus: 18 files, 36 chunks (500 chars, 100 overlap) | Questions: 47 | Candidates per retriever: 10 | Index build: 0.83 s
Embeddings: `sentence-transformers/all-MiniLM-L6-v2` | Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` | RRF k = 60

| Method | Top-1 | Hit@3 | MRR | ms / query |
|---|---|---|---|---|
| dense | 43/47 (91.5%) | 47/47 (100.0%) | 0.957 | 27.6 |
| bm25 | 43/47 (91.5%) | 46/47 (97.9%) | 0.945 | 0.1 |
| hybrid_rrf | 47/47 (100.0%) | 47/47 (100.0%) | 1.000 | 27.8 |
| hybrid_rerank | 47/47 (100.0%) | 47/47 (100.0%) | 1.000 | 78.6 |

### Top-1 by question type

| Method | confusable | direct | keyword | paraphrase |
|---|---|---|---|---|
| dense | 9/9 | 12/12 | 10/12 | 12/14 |
| bm25 | 9/9 | 11/12 | 9/12 | 14/14 |
| hybrid_rrf | 9/9 | 12/12 | 12/12 | 14/14 |
| hybrid_rerank | 9/9 | 12/12 | 12/12 | 14/14 |

### Questions hybrid + rerank did not rank first (0)

