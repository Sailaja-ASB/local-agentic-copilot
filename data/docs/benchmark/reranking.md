# Reranking with Cross-Encoders

Reranking is a second stage. A fast first-stage retriever returns a candidate set, for example the top 10 chunks, and a reranker re-orders only those candidates. A cross-encoder reads the query and one candidate together in a single forward pass, so it can judge relevance far more precisely than comparing two independently computed embeddings.

The price is speed: a cross-encoder must run once per query-candidate pair, which is why it is only applied to a small candidate list and never to the whole corpus. cross-encoder/ms-marco-MiniLM-L-6-v2 is a common lightweight choice. Reranking can fix the order but cannot recover a document the first stage missed.
