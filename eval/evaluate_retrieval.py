import json  # Reads our fixed evaluation questions from questions.json.
from pathlib import Path  # Lets Python work with file paths safely.

import chromadb  # Opens the local Chroma vector database.
from sentence_transformers import SentenceTransformer  # Converts questions into embeddings.


QUESTIONS_PATH = Path("eval/questions.json")  # Location of our predefined evaluation questions.
DB_PATH = "data/chroma_db"  # Location of our existing vector database.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Same embedding model used during ingestion.


questions = json.loads(
    QUESTIONS_PATH.read_text(encoding="utf-8")
)  # Load all benchmark questions and expected source documents.

model = SentenceTransformer(MODEL_NAME)  # Load the embedding model used by our retriever.

client = chromadb.PersistentClient(path=DB_PATH)  # Open our local Chroma database.

collection = client.get_collection(
    name="documents"
)  # Open the collection containing our indexed document chunks.


top1_correct = 0  # Counts how often the correct source is ranked first.
hit3_correct = 0  # Counts how often the correct source appears anywhere in the top 3.


for item in questions:  # Evaluate every benchmark question.

    question = item["question"]  # Read the current test question.

    expected_source = item["expected_source"]  # Read the source file we expect retrieval to find.

    question_embedding = model.encode(question).tolist()  # Convert the question into a semantic vector.

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
    )  # Retrieve the three most semantically similar chunks.

    retrieved_sources = [
        metadata["source"]
        for metadata in results["metadatas"][0]
    ]  # Extract the filenames of the top 3 retrieved results.

    top1_passed = retrieved_sources[0] == expected_source  # Check whether the expected source ranked first.

    hit3_passed = expected_source in retrieved_sources  # Check whether the expected source appeared anywhere in top 3.

    if top1_passed:
        top1_correct += 1  # Increase Top-1 score when the first result is correct.

    if hit3_passed:
        hit3_correct += 1  # Increase Hit@3 score when the correct source is among top 3.

    print(
        f"{'PASS' if top1_passed else 'FAIL'} | "
        f"{question}\n"
        f"Expected: {expected_source}\n"
        f"Top 3: {retrieved_sources}\n"
    )  # Show the ranking returned for each benchmark question.


total = len(questions)  # Number of questions in our benchmark.

top1_accuracy = top1_correct / total * 100  # Calculate Top-1 retrieval accuracy.

hit3_accuracy = hit3_correct / total * 100  # Calculate Hit@3 retrieval accuracy.


print(f"Top-1 Accuracy: {top1_correct}/{total} = {top1_accuracy:.1f}%")  # Show how often Rank 1 was correct.

print(f"Hit@3: {hit3_correct}/{total} = {hit3_accuracy:.1f}%")  # Show how often the correct source appeared within top 3.