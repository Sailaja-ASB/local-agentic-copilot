import json  # Reads our fixed advanced benchmark questions.
from pathlib import Path  # Lets Python work with files and folders.

import chromadb  # Retrieves the initial candidate documents from Chroma.
from sentence_transformers import SentenceTransformer, CrossEncoder  # Dense retriever + second-stage reranker.


QUESTIONS_PATH = Path("eval/advanced_questions.json")  # Same fixed 8-question benchmark.
DB_PATH = "data/chroma_advanced"  # Existing advanced Chroma vector database.

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Same dense model used in our baseline.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Small cross-encoder that scores query-document pairs directly.


embedding_model = SentenceTransformer(EMBEDDING_MODEL)  # Loads our dense retrieval model.

reranker = CrossEncoder(RERANKER_MODEL)  # Loads the more precise second-stage ranking model.

client = chromadb.PersistentClient(path=DB_PATH)  # Opens the existing advanced vector database.

collection = client.get_collection(
    name="advanced_documents"
)  # Opens the indexed advanced documents.


questions = json.loads(
    QUESTIONS_PATH.read_text(encoding="utf-8")
)  # Loads the exact same benchmark used for previous experiments.


top1_correct = 0  # Counts how often reranking puts the expected source first.
hit3_correct = 0  # Counts how often the expected source is present in the candidate set.


for item in questions:  # Evaluate every fixed question.

    question = item["question"]  # Current benchmark question.

    expected_source = item["expected_source"]  # Correct source for this question.


    query_embedding = embedding_model.encode(question).tolist()  # Embed the query for first-stage dense retrieval.

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )  # Retrieve the top 3 candidates; our earlier benchmark showed the correct answer is always here.


    candidate_documents = results["documents"][0]  # Extract the three candidate text chunks.

    candidate_metadata = results["metadatas"][0]  # Extract their source filenames.


    pairs = [
    [
        question,
        f"Source topic: {candidate_metadata[index]['source'].replace('_', ' ').replace('.txt', '')}. "
        f"Document: {document}"
    ]
    for index, document in enumerate(candidate_documents)
]  # Gives the reranker both the source topic and document text so it has more context when distinguishing overlapping candidates.


    reranker_scores = reranker.predict(pairs)  # Score each candidate based on the full query-document relationship.


    ranked_indexes = sorted(
        range(len(reranker_scores)),
        key=lambda index: reranker_scores[index],
        reverse=True,
    )  # Sort candidates from highest reranker score to lowest.


    reranked_sources = [
        candidate_metadata[index]["source"]
        for index in ranked_indexes
    ]  # Convert reranked candidates back into source filenames.


    top1_passed = reranked_sources[0] == expected_source  # Check whether reranking fixed Rank 1.

    hit3_passed = expected_source in reranked_sources  # Confirm the expected source remained in the candidate set.


    top1_correct += int(top1_passed)  # Update reranked Top-1 score.

    hit3_correct += int(hit3_passed)  # Update Hit@3 score.


    print(
        f"{'PASS' if top1_passed else 'FAIL'} | {question}\n"
        f"Expected: {expected_source}\n"
        f"Before rerank: {[m['source'] for m in candidate_metadata]}\n"
        f"After rerank: {reranked_sources}\n"
        f"Scores: {[round(float(score), 4) for score in reranker_scores]}\n"
    )  # Shows exactly whether reranking changed the ordering.


total = len(questions)  # Number of benchmark questions.

top1_accuracy = top1_correct / total * 100  # Calculate reranked Top-1 accuracy.

hit3_accuracy = hit3_correct / total * 100  # Calculate candidate Hit@3.


print(
    f"Reranked Top-1 Accuracy: {top1_correct}/{total} = {top1_accuracy:.1f}%"
)  # Final reranking Top-1 metric.

print(
    f"Reranked Hit@3: {hit3_correct}/{total} = {hit3_accuracy:.1f}%"
)  # Final Hit@3 metric.