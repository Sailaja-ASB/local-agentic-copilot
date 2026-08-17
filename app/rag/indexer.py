from pathlib import Path  # Lets us point to the document directory.

import chromadb  # Stores our document chunks and embeddings.
from sentence_transformers import SentenceTransformer  # Generates semantic embeddings.

from app.rag.ingestion import load_documents  # Uses our tested TXT/MD/PDF ingestion pipeline.


DOCS_DIR = Path("data/index_test")  # Small test dataset used to verify indexing before processing large PDFs.
DB_PATH = "data/chroma_documents"  # Separate production-style document index.

COLLECTION_NAME = "knowledge_base"  # Name of our Chroma collection.

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model used for documents and future queries.

def build_index():
    """Load documents, generate embeddings, and store them in Chroma."""

    records = load_documents(DOCS_DIR)  # Read and chunk TXT, Markdown, and PDF documents.

    if not records:
        raise ValueError("No document chunks found.")  # Prevent building an empty index.

    print(f"Loaded {len(records)} chunks for indexing.")  # Show how much content will be embedded.

    texts = [
        record["text"]
        for record in records
    ]  # Extract chunk text for embedding.

    ids = [
        record["id"]
        for record in records
    ]  # Extract stable chunk identifiers.

    metadatas = [
        {
            "source": record["source"],
            "page": record["page"] if record["page"] is not None else -1,
            "chunk_index": record["chunk_index"],
        }
        for record in records
    ]  # Preserve source, page, and chunk metadata inside Chroma.

    model = SentenceTransformer(MODEL_NAME)  # Load the embedding model.

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
    ).tolist()  # Generate embeddings for all document chunks.

    client = chromadb.PersistentClient(path=DB_PATH)  # Open/create our persistent vector database.

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Remove an old version of this index when it exists.

    collection = client.create_collection(
        name=COLLECTION_NAME
    )  # Create a clean collection for the current document set.

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )  # Store text, vectors, and citation metadata together.

    print(f"Indexed {len(records)} chunks successfully.")  # Confirm index creation.


if __name__ == "__main__":
    build_index()  # Build the index when this file is executed directly.