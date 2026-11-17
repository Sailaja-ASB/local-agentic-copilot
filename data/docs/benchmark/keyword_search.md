# Keyword Search and BM25

Lexical retrieval scores documents by the words they share with the query. BM25 is the standard scoring function: it rewards terms that appear often in a document (term frequency), discounts terms that appear in almost every document (inverse document frequency) and normalises for document length so long documents do not win just by being long.

Keyword search shines on exact matches: product names, stack-trace fragments, HTTP status codes and configuration keys. It fails when the user describes the idea with synonyms that never appear in the text.
