# Hybrid Search with Reciprocal Rank Fusion

Hybrid retrieval runs dense search and BM25 side by side and merges the two ranked lists. Reciprocal Rank Fusion (RRF) is a simple merging rule: every document gets 1 / (k + rank) from each list it appears in, usually with k = 60, and the scores are added. RRF only uses positions, so it needs no score calibration between the very different BM25 and cosine scales.

Hybrid search makes a system robust to both kinds of query: conceptual questions are caught by the dense side and exact identifiers by the lexical side. The fused list is often passed to a reranker afterwards.
