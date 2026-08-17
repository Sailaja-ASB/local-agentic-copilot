from pathlib import Path  # Lets the test point to our sample document folder.

from app.rag.ingestion import load_documents  # Imports the ingestion function we are testing.


def test_load_documents():
    docs_dir = Path("data/docs")  # Folder containing TXT, Markdown, and the local PDF.

    records = load_documents(docs_dir)  # Load and chunk all supported documents.

    assert len(records) > 0  # Confirms ingestion produced chunks.

    assert all("text" in record for record in records)  # Every record must contain text.

    assert all("source" in record for record in records)  # Every record must remember its source file.

    assert all("page" in record for record in records)  # Every record must include a page field.

    pdf_records = [
        record for record in records
        if record["source"].lower().endswith(".pdf")
    ]  # Select only chunks that came from PDFs.

    assert len(pdf_records) > 0  # Confirms our test actually found PDF chunks.

    assert all(
        record["page"] is not None
        for record in pdf_records
    )  # Confirms every PDF chunk remembers the page it came from.

    print(f"Loaded {len(records)} chunks")  # Shows total chunks created.
    print(f"Loaded {len(pdf_records)} PDF chunks")  # Shows how many chunks came from PDFs.
    print(f"First PDF page: {pdf_records[0]['page']}")  # Quick sanity check that page numbering starts correctly.