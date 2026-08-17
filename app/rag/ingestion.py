from pathlib import Path  # Lets us work with files and folders.

from pypdf import PdfReader  # Reads text from PDF files.


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}  # File types our ingestion pipeline supports.

CHUNK_SIZE = 500  # Maximum number of characters in each chunk.

CHUNK_OVERLAP = 100  # Repeats some text between chunks so important context is not cut off at boundaries.


def read_document(file_path: Path) -> str:
    """Read supported document types and return plain text."""

    suffix = file_path.suffix.lower()  # Detect the file extension.

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")  # Read plain text files.

    if suffix == ".md":
        return file_path.read_text(encoding="utf-8")  # Markdown is text, so we can read it directly.

    if suffix == ".pdf":
        reader = PdfReader(file_path)  # Open the PDF.

        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]  # Extract text from every PDF page.

        return "\n".join(pages)  # Combine all page text into one document.

    raise ValueError(
        f"Unsupported file type: {suffix}"
    )  # Stop cleanly if someone gives us an unsupported format.


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks."""

    if not text.strip():
        return []  # Do not create chunks from an empty document.

    chunks = []  # Stores all generated chunks.

    start = 0  # Character position where the current chunk begins.

    while start < len(text):
        end = start + chunk_size  # Calculate where the current chunk should end.

        chunk = text[start:end].strip()  # Extract and clean the current chunk.

        if chunk:
            chunks.append(chunk)  # Keep only non-empty chunks.

        start += chunk_size - overlap  # Move forward while preserving overlap with the previous chunk.

    return chunks  # Return all generated chunks.


def load_documents(directory: Path) -> list[dict]:
    """Load and chunk every supported document in a directory."""

    records = []  # Stores chunks plus metadata.

    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue  # Ignore folders.

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue  # Ignore unsupported file types.

        text = read_document(file_path)  # Extract readable text.

        chunks = chunk_text(text)  # Split the document into overlapping pieces.

        for index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{file_path.stem}-chunk-{index}",
                    "text": chunk,
                    "source": file_path.name,
                    "chunk_index": index,
                }
            )  # Save each chunk with traceable metadata.

    return records  # Return everything ready for embedding/indexing.