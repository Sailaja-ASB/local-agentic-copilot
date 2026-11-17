# Dense Vector Search

Dense retrieval turns every query and every document chunk into an embedding, a list of a few hundred numbers that captures meaning. Chunks are ranked by cosine similarity or distance between the query vector and each chunk vector. Because it compares meaning instead of words, dense search can match a question that uses completely different wording from the source, for example "car" and "automobile".

Its weakness is precision on rare strings. Exact identifiers such as error codes, version numbers, environment variable names or function names carry little semantic signal, so the embedding of a query containing them may land near many unrelated chunks.
