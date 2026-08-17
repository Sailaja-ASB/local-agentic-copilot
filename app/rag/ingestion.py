from pathlib import Path  # Lets us work with files and folders.

from pypdf import PdfReader  # Reads text from PDF files.


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}  # File types our ingestion pipeline supports.

CHUNK_SIZE = 500  # Maximum number of characters in each chunk.

CHUNK_OVERLAP = 100  # Repeats some text between chunks so context is not cut off abruptly.


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks."""

    if not text.strip():
        return []  # Ignore empty text.

    chunks = []  # Stores the chunks we create.

    start = 0  # Starting character position.

    while start < len(text):
        end = start + chunk_size  # Calculate where the chunk should end.

        chunk = text[start:end].strip()  # Extract and clean the chunk.

        if chunk:
            chunks.append(chunk)  # Keep non-empty chunks only.

        start += chunk_size - overlap  # Move forward while preserving overlap.

    return chunks  # Return all chunks.


def load_documents(directory: Path) -> list[dict]:
    """Load and chunk supported documents while preserving source metadata."""

    records = []  # Stores every chunk plus metadata.

    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue  # Ignore folders.

        suffix = file_path.suffix.lower()  # Detect file type.

        if suffix not in SUPPORTED_EXTENSIONS:
            continue  # Ignore unsupported formats.


        if suffix in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8")  # Read text/Markdown normally.

            chunks = chunk_text(text)  # Split into overlapping chunks.

            for index, chunk in enumerate(chunks):
                records.append(
                    {
                        "id": f"{file_path.stem}-chunk-{index}",
                        "text": chunk,
                        "source": file_path.name,
                        "page": None,
                        "chunk_index": index,
                    }
                )  # Save chunk metadata; TXT/MD files do not have page numbers.


        elif suffix == ".pdf":
            reader = PdfReader(file_path)  # Open the PDF.

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""  # Extract text from this specific page.

                chunks = chunk_text(text)  # Chunk one page at a time instead of flattening the whole PDF.

                for index, chunk in enumerate(chunks):
                    records.append(
                        {
                            "id": f"{file_path.stem}-page-{page_number}-chunk-{index}",
                            "text": chunk,
                            "source": file_path.name,
                            "page": page_number,
                            "chunk_index": index,
                        }
                    )  # Preserve the exact PDF page number for future citations.

    return records  # Return all chunks ready for indexing.