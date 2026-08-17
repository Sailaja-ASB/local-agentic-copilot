import json  # Reads our fixed advanced benchmark questions.
from pathlib import Path  # Lets Python work with folders and files.

import chromadb  # Gives us dense semantic retrieval through Chroma.
from rank_bm25 import BM25Okapi  # Adds lexical BM25 keyword retrieval.
from sentence_transformers import SentenceTransformer  # Creates dense semantic embeddings.


DOCS_DIR = Path("data/docs/advanced")  # Folder containing our harder overlapping documents.
QUESTIONS_PATH = Path("eval/advanced_questions.json")  # Fixed benchmark we will NOT change.
DB_PATH = "data/chroma_advanced"  # Existing dense vector database.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Same dense embedding model as before.


documents = []  # Stores the full text of each source document.
sources = []  # Stores the filename corresponding to each document.


for file_path in DOCS_DIR.glob("*.txt"):  # Read every advanced document.

    text = file_path.read_text(encoding="utf-8")  # Load document text.

    documents.append(text)  # Add text to our searchable corpus.

    sources.append(file_path.name)  # Remember which filename owns this text.


tokenized_documents = [
    document.lower().split()
    for document in documents
]  # Lowercase and split documents into tokens because BM25 needs tokenized text.


bm25 = BM25Okapi(tokenized_documents)  # Build the lexical BM25 search index.


model = SentenceTransformer(MODEL_NAME)  # Load our dense embedding model.

client = chromadb.PersistentClient(path=DB_PATH)  # Open our existing dense Chroma database.

collection = client.get_collection(
    name="advanced_documents"
)  # Open the advanced dense retrieval collection.


questions = json.loads(
    QUESTIONS_PATH.read_text(encoding="utf-8")
)  # Load the exact same eight benchmark questions.


top1_correct = 0  # Counts how often hybrid retrieval ranks the correct source first.
hit3_correct = 0  # Counts how often the correct source appears in hybrid Top 3.


for item in questions:  # Evaluate every fixed benchmark question.

    question = item["question"]  # Current question.

    expected_source = item["expected_source"]  # Correct document for this question.


    # ---------- DENSE RETRIEVAL ----------

    query_embedding = model.encode(question).tolist()  # Convert question into a dense semantic vector.

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(documents),
    )  # Retrieve all documents ranked by semantic similarity.

    dense_sources = [
        metadata["source"]
        for metadata in dense_results["metadatas"][0]
    ]  # Extract dense ranking as source filenames.


    # ---------- BM25 RETRIEVAL ----------

    tokenized_query = question.lower().split()  # Tokenize query using the same simple preprocessing as the documents.

    bm25_scores = bm25.get_scores(tokenized_query)  # Calculate lexical relevance score for every document.

    bm25_ranked_indexes = sorted(
        range(len(bm25_scores)),
        key=lambda index: bm25_scores[index],
        reverse=True,
    )  # Sort document indexes from highest BM25 score to lowest.

    bm25_sources = [
        sources[index]
        for index in bm25_ranked_indexes
    ]  # Convert BM25 ranking indexes into filenames.


    # ---------- RECIPROCAL RANK FUSION ----------

    fusion_scores = {}  # Stores one combined score for each source.

    for rank, source in enumerate(dense_sources, start=1):  # Add dense ranking contribution.

        fusion_scores[source] = fusion_scores.get(source, 0) + 1 / (60 + rank)

    for rank, source in enumerate(bm25_sources, start=1):  # Add lexical ranking contribution.

        fusion_scores[source] = fusion_scores.get(source, 0) + 1 / (60 + rank)

    hybrid_sources = sorted(
        fusion_scores,
        key=fusion_scores.get,
        reverse=True,
    )  # Sort sources by their combined dense + BM25 score.


    top3 = hybrid_sources[:3]  # Keep the three highest-ranked hybrid results.

    top1_passed = top3[0] == expected_source  # Check whether correct source ranked first.

    hit3_passed = expected_source in top3  # Check whether correct source appeared in Top 3.


    top1_correct += int(top1_passed)  # Update Top-1 score.

    hit3_correct += int(hit3_passed)  # Update Hit@3 score.


    print(
        f"{'PASS' if top1_passed else 'FAIL'} | {question}\n"
        f"Expected: {expected_source}\n"
        f"Dense: {dense_sources[:3]}\n"
        f"BM25: {bm25_sources[:3]}\n"
        f"Hybrid: {top3}\n"
    )  # Show all three rankings so we can understand why fusion helped or failed.


total = len(questions)  # Number of questions in the advanced benchmark.

top1_accuracy = top1_correct / total * 100  # Calculate hybrid Top-1 accuracy.

hit3_accuracy = hit3_correct / total * 100  # Calculate hybrid Hit@3.


print(
    f"Hybrid Top-1 Accuracy: {top1_correct}/{total} = {top1_accuracy:.1f}%"
)  # Print final hybrid Top-1 metric.

print(
    f"Hybrid Hit@3: {hit3_correct}/{total} = {hit3_accuracy:.1f}%"
)  # Print final hybrid Hit@3 metric.