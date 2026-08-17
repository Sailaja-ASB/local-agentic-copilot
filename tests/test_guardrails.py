from app.agents.guardrails import validate_solution  # Imports the deterministic guardrail logic we want to test.


def test_guardrails_reject_unsupported_stack():
    """Guardrails should reject unsupported libraries and missing required stack components."""

    task = (
        "Explain how page-aware ingestion should support citations "
        "in a RAG application."
    )

    bad_solution = """
import pdfplumber
from langchain.vectorstores import Chroma

Use pdfplumber to read the PDF.
Store page numbers later.
"""

    result = validate_solution(
        task,
        bad_solution,
    )

    assert result["passed"] is False  # The bad solution must not pass.

    assert len(result["violations"]) > 0  # At least one violation should be reported.

    violation_text = " ".join(result["violations"]).lower()

    assert "pdfplumber" in violation_text  # Confirms unsupported PDF parser is detected.

    assert "pypdf" in violation_text  # Confirms the expected PDF stack is required.

    assert "chromadb" in violation_text  # Confirms page metadata storage in ChromaDB is required.


def test_guardrails_accept_correct_stack():
    """Guardrails should accept a solution that follows the existing project stack."""

    task = (
        "Explain how page-aware ingestion should support citations "
        "in a RAG application."
    )

    good_solution = """
Use pypdf PdfReader to read each PDF page.

Preserve the original page number as metadata for every chunk.

Generate embeddings using sentence-transformers.

Store each chunk in chromadb with metadata containing:
- source filename
- page number
- chunk index

During retrieval, return the page metadata with the matching chunk so
the final answer can cite the original PDF page.
"""

    result = validate_solution(
        task,
        good_solution,
    )

    assert result["passed"] is True  # A correct implementation should pass deterministic checks.

    assert result["violations"] == []  # No violations should remain.