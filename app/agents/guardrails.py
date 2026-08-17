import re  # Lets us detect actual code/library usage patterns more precisely.
from typing import Dict, List  # Provides clear type hints for guardrail results.


PROJECT_STACK = {
    "pdf_parser": "pypdf",
    "embeddings": "sentence-transformers",
    "vector_db": "chromadb",
    "llm_runtime": "ollama",
    "llm_model": "qwen3:4b",
}  # Defines the technologies that our project actually uses.


DISALLOWED_USAGE_PATTERNS = {
    "pdfplumber": [
        r"\bimport\s+pdfplumber\b",
        r"\bfrom\s+pdfplumber\b",
        r"\bpdfplumber\.open\s*\(",
    ],
    "pymupdf": [
        r"\bimport\s+pymupdf\b",
        r"\bfrom\s+pymupdf\b",
        r"\bimport\s+fitz\b",
        r"\bfrom\s+fitz\b",
        r"\bfitz\.open\s*\(",
    ],
    "langchain": [
        r"\bimport\s+langchain\b",
        r"\bfrom\s+langchain\b",
    ],
    "huggingfaceembeddings": [
        r"\bHuggingFaceEmbeddings\s*\(",
        r"\bfrom\s+.*HuggingFaceEmbeddings\b",
        r"\bimport\s+HuggingFaceEmbeddings\b",
    ],
}  # Detects actual attempts to use unsupported technologies instead of merely mentioning their names.


def _uses_disallowed_technology(
    solution: str,
    patterns: List[str],
) -> bool:
    """Return True when the solution actually uses a prohibited technology."""

    for pattern in patterns:
        if re.search(
            pattern,
            solution,
            flags=re.IGNORECASE,
        ):
            return True  # Reject only when an actual usage pattern is detected.

    return False


def validate_solution(
    user_task: str,
    proposed_solution: str,
) -> Dict[str, object]:
    """Apply deterministic project-stack checks before LLM review."""

    violations: List[str] = []  # Stores deterministic problems found in the solution.

    normalized_solution = proposed_solution.lower()  # Makes semantic checks case-insensitive.


    # ---------------------------------------------------------
    # CHECK 1: UNSUPPORTED LIBRARY USAGE
    # ---------------------------------------------------------

    for technology, patterns in DISALLOWED_USAGE_PATTERNS.items():

        if _uses_disallowed_technology(
            proposed_solution,
            patterns,
        ):

            violations.append(
                f"Unsupported implementation detected: {technology}. "
                "Use the existing project stack instead."
            )  # Reject actual unsupported implementation choices.


    # ---------------------------------------------------------
    # CHECK 2: PAGE-AWARE CITATION REQUIREMENTS
    # ---------------------------------------------------------

    citation_task = (
        "page-aware" in user_task.lower()
        or "citation" in user_task.lower()
    )  # Detect whether this task requires page-aware citation behavior.


    if citation_task:

        if "page" not in normalized_solution:
            violations.append(
                "The solution must explain preservation of PDF page metadata."
            )  # Page-aware tasks must discuss page metadata.


        if "pypdf" not in normalized_solution:
            violations.append(
                "The implementation must use the existing pypdf PDF ingestion stack."
            )  # Require our actual PDF parser.


        if "chromadb" not in normalized_solution:
            violations.append(
                "The solution must explain how page metadata is stored with chunks in ChromaDB."
            )  # Require metadata persistence in our actual vector database.


    # ---------------------------------------------------------
    # CHECK 3: BASIC PDF IMPLEMENTATION SANITY
    # ---------------------------------------------------------

    suspicious_plain_pdf_open = re.search(
        r'open\s*\(\s*[^)]*\.pdf[^)]*["\']r["\']',
        proposed_solution,
        flags=re.IGNORECASE,
    )

    if suspicious_plain_pdf_open:
        violations.append(
            "PDFs should not be parsed as plain text with open(..., 'r'); use pypdf PdfReader."
        )  # Prevent the fake PDF-reading implementation we saw earlier.


    passed = len(violations) == 0  # Approval requires zero deterministic violations.


    return {
        "passed": passed,
        "violations": violations,
        "project_stack": PROJECT_STACK,
    }


if __name__ == "__main__":

    test_task = (
        "Explain how page-aware ingestion should be used "
        "to support citations in a RAG application."
    )


    intentionally_bad_solution = """
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    pass
"""


    result = validate_solution(
        test_task,
        intentionally_bad_solution,
    )


    print("\nGUARDRAIL PASSED:")
    print(result["passed"])


    print("\nVIOLATIONS:")

    for violation in result["violations"]:
        print(f"- {violation}")