from pathlib import Path  # Lets Python work with project file/folder paths.

import chromadb  # Local vector database for storing and searching document embeddings.
from sentence_transformers import SentenceTransformer  # Converts text into semantic embeddings.


DOCS_DIR = Path("data/docs")  # Folder containing all documents our RAG system should ingest.

DB_PATH = "data/chroma_db"  # Folder where Chroma permanently stores the vector database.

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Small embedding model for semantic retrieval.

CHUNK_SIZE = 500  # Maximum number of characters in each document chunk.


model = SentenceTransformer(MODEL_NAME)  # Load the embedding model into memory.

client = chromadb.PersistentClient(path=DB_PATH)  # Open/create our local Chroma database.

collection = client.get_or_create_collection(
    name="documents"
)  # Create or reopen the collection containing our searchable document chunks.


all_chunks = []  # Stores every text chunk from all documents.

all_ids = []  # Stores a unique ID for every chunk.

all_metadata = []  # Stores information such as which source file each chunk came from.


for file_path in DOCS_DIR.glob("*.txt"):  # Find every .txt document inside data/docs.

    text = file_path.read_text(encoding="utf-8")  # Read the current document into Python.

    chunks = [
        text[i:i + CHUNK_SIZE]
        for i in range(0, len(text), CHUNK_SIZE)
    ]  # Split the current document into smaller searchable chunks.

    for index, chunk in enumerate(chunks):  # Process every chunk from this document.

        chunk_id = f"{file_path.stem}-chunk-{index}"  # Create a unique ID using filename + chunk number.

        all_chunks.append(chunk)  # Add this chunk to our complete document collection.

        all_ids.append(chunk_id)  # Save the unique ID for Chroma.

        all_metadata.append(
            {
                "source": file_path.name,
                "chunk": index,
            }
        )  # Remember exactly which document and chunk produced this text.


embeddings = model.encode(all_chunks).tolist()  # Convert every document chunk into semantic vectors.


collection.upsert(
    ids=all_ids,  # Store unique identifiers for every chunk.
    documents=all_chunks,  # Store the readable chunk text.
    embeddings=embeddings,  # Store vectors used for semantic similarity search.
    metadatas=all_metadata,  # Store source filenames so we can later provide citations.
)  # Add/update all document chunks inside Chroma.


question = "What does RAG combine?"  # Test whether the retriever chooses the RAG document instead of Docker/FastAPI.

question_embedding = model.encode(question).tolist()  # Convert our question into the same vector space.


results = collection.query(
    query_embeddings=[question_embedding],  # Search for chunks closest in meaning to the question.
    n_results=2,  # Retrieve the two most relevant chunks so we can inspect ranking.
)


print("\nQUESTION:")
print(question)  # Show the question being tested.

print("\nTOP RETRIEVAL RESULTS:")

for rank, document in enumerate(results["documents"][0], start=1):  # Display each retrieved result in ranked order.

    metadata = results["metadatas"][0][rank - 1]  # Get the matching source information.

    distance = results["distances"][0][rank - 1]  # Get Chroma's distance score for this match.

    print(f"\nRank {rank}")  # Show retrieval ranking.

    print(f"Source: {metadata['source']}")  # Show which document produced the result.

    print(f"Distance: {distance:.4f}")  # Show similarity distance; smaller usually means closer.

    print(f"Text: {document}")  # Show the actual retrieved evidence.