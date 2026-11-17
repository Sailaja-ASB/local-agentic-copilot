# App search benchmark (app/rag/search.py through Chroma)

Run: 2026-11-16 19:22 | Questions: 47 | top_k = 10 chunks per query

| Mode | Top-1 | Hit@3 | MRR | dense ms | BM25 ms | rerank ms | total ms |
|---|---|---|---|---|---|---|---|
| dense | 43/47 (91.5%) | 47/47 (100.0%) | 0.957 | 14.7 | - | - | 14.7 |
| bm25 | 43/47 (91.5%) | 46/47 (97.9%) | 0.945 | - | 0.1 | - | 0.1 |
| hybrid | 47/47 (100.0%) | 47/47 (100.0%) | 1.000 | 9.2 | 0.1 | - | 9.3 |
| hybrid_rerank | 47/47 (100.0%) | 47/47 (100.0%) | 1.000 | 7.9 | 0.1 | 49.7 | 57.7 |

### Not ranked first

**dense** (4)
- q05 rank=2 expected=embedding_models.md got=['vector_search.md', 'embedding_models.md', 'reranking.md']
- q14 rank=2 expected=reranking.md got=['embedding_models.md', 'reranking.md', 'chunking.md']
- q33 rank=2 expected=load_shedding.md got=['rate_limiting.md', 'load_shedding.md', 'local_models_ollama.md']
- q42 rank=2 expected=rate_limiting.md got=['load_shedding.md', 'rate_limiting.md', 'circuit_breaker.md']

**bm25** (4)
- q03 rank=3 expected=vector_search.md got=['chunking.md', 'embedding_models.md', 'vector_search.md']
- q04 rank=4 expected=embedding_models.md got=['rate_limiting.md', 'rag_pipeline.md', 'vector_search.md']
- q33 rank=2 expected=load_shedding.md got=['prompt_injection.md', 'load_shedding.md', 'autoscaling_hpa.md']
- q40 rank=3 expected=circuit_breaker.md got=['prompt_injection.md', 'retrieval_metrics.md', 'circuit_breaker.md']

**hybrid** (0)

**hybrid_rerank** (0)
