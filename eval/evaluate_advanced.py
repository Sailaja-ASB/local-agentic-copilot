import json  # Reads our predefined advanced benchmark questions.
from pathlib import Path  # Lets Python work safely with files and folders.

import chromadb  # Stores and searches our advanced document embeddings.
from sentence_transformers import SentenceTransformer  # Converts documents and questions into semantic vectors.


DOCS_DIR = Path("data/docs/advanced")  # Folder containing only our harder overlapping documents.
QUESTIONS_PATH = Path("eval/advanced_questions.json")  # Fixed advanced benchmark written before testing.
DB_PATH = "data/chroma_advanced"  # Separate vector database so we don't modify our baseline database.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Same dense embedding model used in our baseline.
CHUNK_SIZE = 500  # Split longer documents into searchable pieces.


model = SentenceTransformer(MODEL_NAME)  # Load our dense embedding model.

client = chromadb.PersistentClient(path=DB_PATH)  # Create/open a separate database for this experiment.

collection = client.get_or_create_collection(
    name="advanced_documents"
)  # Create the collection that stores our harder document chunks.


all_chunks = []  # Holds every text chunk we will index.
all_ids = []  # Holds a unique ID for every chunk.
all_metadata = []  # Remembers which source document each chunk came from.


for file_path in DOCS_DIR.glob("*.txt"):  # Read every advanced test document.

    text = file_path.read_text(encoding="utf-8")  # Load the document text.

    chunks = [
        text[i:i + CHUNK_SIZE]
        for i in range(0, len(text), CHUNK_SIZE)
    ]  # Split each document into chunks.

    for index, chunk in enumerate(chunks):  # Process each chunk separately.

        all_chunks.append(chunk)  # Add the chunk to our indexing list.

        all_ids.append(
            f"{file_path.stem}-chunk-{index}"
        )  # Give the chunk a unique ID.

        all_metadata.append(
            {"source": file_path.name}
        )  # Save its original filename for evaluation.


embeddings = model.encode(all_chunks).tolist()  # Convert all chunks into dense semantic vectors.

collection.upsert(
    ids=all_ids,
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadata,
)  # Store the advanced dataset in Chroma.


questions = json.loads(
    QUESTIONS_PATH.read_text(encoding="utf-8")
)  # Load our fixed benchmark questions.


top1_correct = 0  # Count questions where the correct source ranks first.
hit3_correct = 0  # Count questions where the correct source appears in the top 3.


for item in questions:  # Evaluate every question without changing the benchmark.

    question = item["question"]  # Get the current question.
    expected_source = item["expected_source"]  # Get its correct source.

    query_embedding = model.encode(question).tolist()  # Embed the question for semantic search.

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )  # Retrieve the three nearest chunks.

    retrieved_sources = [
        metadata["source"]
        for metadata in results["metadatas"][0]
    ]  # Extract the ranked source filenames.

    top1_passed = retrieved_sources[0] == expected_source  # Check Rank 1 correctness.
    hit3_passed = expected_source in retrieved_sources  # Check whether correct source appears in top 3.

    top1_correct += int(top1_passed)  # Update Top-1 score.
    hit3_correct += int(hit3_passed)  # Update Hit@3 score.

    print(
        f"{'PASS' if top1_passed else 'FAIL'} | {question}\n"
        f"Expected: {expected_source}\n"
        f"Top 3: {retrieved_sources}\n"
    )  # Display every result so we can inspect failures.


total = len(questions)  # Count the total benchmark questions.

print(
    f"Top-1 Accuracy: {top1_correct}/{total} = "
    f"{top1_correct / total * 100:.1f}%"
)  # Calculate our harder Top-1 baseline.

print(
    f"Hit@3: {hit3_correct}/{total} = "
    f"{hit3_correct / total * 100:.1f}%"
)  # Calculate our harder Hit@3 baseline.