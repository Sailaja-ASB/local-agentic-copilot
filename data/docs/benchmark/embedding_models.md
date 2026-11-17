# Embedding Models

An embedding model maps text to a fixed-size vector. all-MiniLM-L6-v2 is a small sentence-transformers model that produces 384-dimensional vectors and runs comfortably on a laptop CPU. Larger models such as bge-large or e5-large usually retrieve better but are slower and need more memory.

The same model must be used for indexing documents and for encoding queries; mixing models puts vectors in different spaces and makes similarity scores meaningless. Changing the embedding model therefore means re-embedding and re-indexing the whole corpus.
