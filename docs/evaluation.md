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

## Advanced Dense Retrieval Baseline

### Dataset
- 4 overlapping technical documents
- Topics: vector search, keyword/BM25 search, hybrid search, reranking
- 8 evaluation questions
- Questions were defined before observing retrieval results

### Results
- Top-1 Accuracy: 6/8 = 75%
- Hit@3: 8/8 = 100%

### Failure Analysis
Two questions whose expected source was `vector_search.txt` were ranked below `hybrid_search.txt`.

The correct source was still present within the Top 3 for every question.

### Hypothesis
Dense semantic retrieval successfully finds relevant candidates but can struggle to rank highly overlapping documents correctly.

Next experiments will test whether hybrid lexical + dense retrieval and reranking improve Top-1 accuracy while keeping the same fixed benchmark.

## Hybrid Retrieval Experiment

### Method
Combined dense semantic retrieval and BM25 lexical retrieval using Reciprocal Rank Fusion.

### Results
- Hybrid Top-1 Accuracy: 6/8 = 75%
- Hybrid Hit@3: 8/8 = 100%

### Comparison
Dense baseline:
- Top-1: 75%
- Hit@3: 100%

Hybrid retrieval:
- Top-1: 75%
- Hit@3: 100%

### Interpretation
Adding BM25 with Reciprocal Rank Fusion did not improve Top-1 ranking on the current benchmark.

The correct source remained available within the Top 3 for every query, suggesting that candidate generation is working well but final ranking remains the main weakness.

### Next Hypothesis
A second-stage reranker may improve Top-1 accuracy by comparing the query and candidate documents more directly.


## Cross-Encoder Reranking Experiment

### Method
Dense retrieval first returns the Top-3 candidate chunks. A cross-encoder then scores each query-document pair and reranks the candidate set.

### Results
- Reranked Top-1 Accuracy: 7/8 = 87.5%
- Reranked Hit@3: 8/8 = 100%

### Comparison
- Dense retrieval: 75% Top-1
- Hybrid Dense + BM25: 75% Top-1
- Dense + Cross-Encoder reranking: 87.5% Top-1

### Interpretation
Cross-encoder reranking improved Top-1 accuracy by 12.5 percentage points over the dense baseline.

The candidate-generation stage maintained 100% Hit@3, while the reranker improved final ordering.

One benchmark case remains incorrect and requires failure analysis before further tuning.


### Remaining Failure Analysis

Question:
"Which retrieval approach may struggle with exact identifiers and uncommon technical terms?"

Expected source:
`vector_search.txt`

Reranked result:
1. `keyword_search.txt`
2. `vector_search.txt`
3. `hybrid_search.txt`

Interpretation:
The reranker appears to over-weight the lexical overlap around "exact identifiers" and "uncommon technical terms." Although `vector_search.txt` explicitly states that dense retrieval can struggle with these cases, `keyword_search.txt` describes those same concepts as strengths of lexical retrieval.

This suggests the remaining error is caused by semantic overlap between documents rather than candidate-generation failure.


## Metadata-Aware Reranking

### Method
Added source-topic metadata to each candidate before cross-encoder reranking.

### Results
- Reranked Top-1 Accuracy: 8/8 = 100%
- Reranked Hit@3: 8/8 = 100%

### Comparison
- Dense retrieval: 75%
- Hybrid Dense + BM25: 75%
- Cross-Encoder reranking: 87.5%
- Metadata-aware Cross-Encoder reranking: 100%

### Interpretation
The remaining failure was caused by semantic overlap between `vector_search.txt` and `keyword_search.txt`.

Providing the reranker with explicit source-topic metadata improved disambiguation and corrected the final ranking without modifying the fixed evaluation benchmark.

### Limitation
This benchmark is still small and synthetic, so the 100% result should not be interpreted as production-level accuracy.