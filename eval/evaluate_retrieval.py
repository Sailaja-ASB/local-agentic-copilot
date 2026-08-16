import json  # Reads our fixed evaluation questions from questions.json.
from pathlib import Path  # Lets Python work with file paths safely.

import chromadb  # Opens the local Chroma vector database.
from sentence_transformers import SentenceTransformer  # Converts questions into embeddings.


QUESTIONS_PATH = Path("eval/questions.json")  # Location of our predefined evaluation questions.
DB_PATH = "data/chroma_db"  # Location of the vector database created by our retriever.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Must match the embedding model used during ingestion.


questions = json.loads(
    QUESTIONS_PATH.read_text(encoding="utf-8")
)  # Load all benchmark questions and expected source documents.

model = SentenceTransformer(MODEL_NAME)  # Load the same embedding model used for the documents.

client = chromadb.PersistentClient(path=DB_PATH)  # Open our existing Chroma database.

collection = client.get_collection(
    name="documents"
)  # Open the document collection created by our retriever.


correct = 0  # Counts how many questions retrieved the expected source document.


for item in questions:  # Test every predefined question one at a time.

    question = item["question"]  # Read the benchmark question.

    expected_source = item["expected_source"]  # Read which document should rank first.

    question_embedding = model.encode(question).tolist()  # Convert the question into a semantic vector.

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=1,
    )  # Retrieve only the highest-ranked document chunk.

    predicted_source = results["metadatas"][0][0]["source"]  # Get the filename returned at Rank 1.

    passed = predicted_source == expected_source  # Check whether retrieval selected the correct document.

    if passed:
        correct += 1  # Increase our correct-answer count when retrieval succeeds.

    print(
        f"{'PASS' if passed else 'FAIL'} | "
        f"{question} | "
        f"Expected: {expected_source} | "
        f"Retrieved: {predicted_source}"
    )  # Show the result of every evaluation question.


accuracy = correct / len(questions) * 100  # Calculate overall retrieval accuracy as a percentage.

print(f"\nRetrieval Accuracy: {correct}/{len(questions)} = {accuracy:.1f}%")  # Print our first measurable RAG metric.