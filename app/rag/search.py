import chromadb  # Opens our local Chroma vector database.
from sentence_transformers import SentenceTransformer  # Converts the user's question into an embedding.


DB_PATH = "data/chroma_documents"  # Location of the vector database built by indexer.py.

COLLECTION_NAME = "knowledge_base"  # Name of the Chroma collection containing our indexed chunks.

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Same embedding model used during indexing.


model = SentenceTransformer(MODEL_NAME)  # Load the query embedding model.

client = chromadb.PersistentClient(path=DB_PATH)  # Open our existing Chroma database.

collection = client.get_collection(
    name=COLLECTION_NAME
)  # Open the indexed knowledge-base collection.


def search_documents(question: str, top_k: int = 3) -> list[dict]:
    """Retrieve the most relevant document chunks for a question."""

    query_embedding = model.encode(question).tolist()  # Convert the question into a semantic vector.

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )  # Retrieve the top-k most relevant document chunks.

    matches = []  # Store clean search results for the rest of our application.

    for index in range(len(results["documents"][0])):
        metadata = results["metadatas"][0][index]  # Read source/page metadata for this chunk.

        page = metadata.get("page", -1)  # Get PDF page number; TXT/Markdown uses -1.

        matches.append(
            {
                "text": results["documents"][0][index],
                "source": metadata["source"],
                "page": None if page == -1 else page,
                "chunk_index": metadata["chunk_index"],
                "distance": results["distances"][0][index],
            }
        )  # Return text plus enough metadata to build citations later.

    return matches  # Give the ranked results back to the caller.


if __name__ == "__main__":
    question = "What does RAG combine?"  # Temporary question for our first search-module test.

    matches = search_documents(question, top_k=3)  # Retrieve the three best chunks.

    print(f"\nQUESTION: {question}")  # Show the test question.

    for rank, match in enumerate(matches, start=1):
        print(f"\nRank {rank}")  # Show retrieval order.
        print(f"Source: {match['source']}")  # Show which document supplied this chunk.
        print(f"Page: {match['page']}")  # Show PDF page number when available.
        print(f"Distance: {match['distance']:.4f}")  # Show semantic distance for debugging.
        print(f"Text: {match['text']}")  # Show the actual retrieved evidence.