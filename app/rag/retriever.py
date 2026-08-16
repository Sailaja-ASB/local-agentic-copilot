from pathlib import Path  # Lets Python work with file and folder paths.

import chromadb  # Local vector database used to store and search embeddings.
from sentence_transformers import SentenceTransformer  # Converts text into numerical embeddings.


# ---------- CONFIGURATION ----------

DOC_PATH = Path("data/docs/test.txt")  # Path to the sample document we want our RAG system to search.

DB_PATH = "data/chroma_db"  # Folder where Chroma will permanently store our vector database.

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Small embedding model that converts text into meaning-based vectors.


# ---------- READ DOCUMENT ----------

text = DOC_PATH.read_text(encoding="utf-8")  # Read the entire test document into Python.


# ---------- CHUNK DOCUMENT ----------

chunks = [text[i:i + 500] for i in range(0, len(text), 500)]  # Split the document into 500-character pieces for more precise retrieval.


# ---------- CREATE EMBEDDINGS ----------

model = SentenceTransformer(MODEL_NAME)  # Load our embedding model into memory.

embeddings = model.encode(chunks).tolist()  # Convert every text chunk into a numerical vector representing its meaning.


# ---------- CREATE VECTOR DATABASE ----------

client = chromadb.PersistentClient(path=DB_PATH)  # Create/open our local persistent Chroma vector database.

collection = client.get_or_create_collection(
    name="documents"
)  # Create the collection that will contain our document chunks.


# ---------- STORE DOCUMENTS ----------

ids = [f"chunk-{i}" for i in range(len(chunks))]  # Give every chunk a unique ID.

collection.upsert(
    ids=ids,  # Tell Chroma which unique IDs belong to the chunks.
    documents=chunks,  # Store the original readable text.
    embeddings=embeddings,  # Store the numerical embeddings used for semantic search.
)


# ---------- TEST QUESTION ----------

question = "What does RAG do?"  # Temporary question used to test whether our retrieval system works.


# ---------- EMBED QUESTION ----------

question_embedding = model.encode(question).tolist()  # Convert the question into the same vector format as our documents.


# ---------- SEARCH VECTOR DATABASE ----------

results = collection.query(
    query_embeddings=[question_embedding],  # Search for document chunks semantically similar to our question.
    n_results=1,  # Return the single most relevant chunk for this first test.
)


# ---------- SHOW RESULT ----------

print("\nQUESTION:")  # Print a label so our terminal output is easy to read.
print(question)  # Show the question we searched for.

print("\nRETRIEVED DOCUMENT:")  # Print a label for the retrieved information.
print(results["documents"][0][0])  # Display the document chunk Chroma considered most relevant.