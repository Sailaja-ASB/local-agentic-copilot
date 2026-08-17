from pathlib import Path  # Lets the test point to our sample documents.

from app.rag.ingestion import load_documents  # Imports the ingestion function we just built.


def test_load_documents():
    docs_dir = Path("data/docs")  # Uses our existing sample document folder.

    records = load_documents(docs_dir)  # Loads and chunks all supported documents.

    assert len(records) > 0  # Confirms ingestion produced at least one chunk.

    assert all("text" in record for record in records)  # Confirms every chunk contains text.

    assert all("source" in record for record in records)  # Confirms every chunk remembers its source file.

    print(f"Loaded {len(records)} chunks")  # Shows how many chunks were created.