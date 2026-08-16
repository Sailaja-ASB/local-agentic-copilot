# Retrieval Evaluation

## Baseline Evaluation

### Goal
Measure whether the retriever selects the correct source document for a predefined set of questions.

### Setup
- Documents: 3
- Evaluation questions: 6
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector database: ChromaDB
- Retrieval type: Dense semantic retrieval
- Metrics: Top-1 Accuracy and Hit@3

### Results
- Top-1 Accuracy: 6/6 = 100%
- Hit@3: 6/6 = 100%

### Interpretation
The retriever correctly ranked the expected source document first for all six baseline questions.

This is only an initial sanity-check benchmark using a small and simple dataset. The 100% result should not be interpreted as production-level retrieval accuracy.

### Next Evaluation Steps
- Add more documents.
- Add longer documents.
- Add overlapping topics.
- Add ambiguous questions.
- Add paraphrased questions whose wording differs from the source text.
- Compare baseline dense retrieval with improved retrieval methods such as hybrid search and reranking.