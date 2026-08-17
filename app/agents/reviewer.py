import ollama  # Lets the Reviewer Agent use our local Qwen model.


MODEL_NAME = "qwen3:4b"  # Local model used by the Reviewer Agent.


def review_solution(
    user_task: str,
    research_findings: str,
    proposed_solution: str,
) -> dict:
    """Review a proposed solution for correctness, grounding, and project consistency."""

    prompt = f"""
You are the Reviewer Agent in a technical AI system.

Your job is to evaluate the Coder Agent's proposed solution.

Review criteria:
1. Does the solution actually answer the user's task?
2. Is it grounded in the research findings?
3. Does it invent unsupported facts, libraries, or requirements?
4. Is it technically correct?
5. Does it preserve relevant citations?
6. Does it stay consistent with the existing project stack when that information is available?

Important project stack:
- PDF parsing: pypdf
- Embeddings: sentence-transformers
- Vector database: chromadb
- Local LLM runtime: Ollama
- Local model: qwen3:4b
- Current project modules:
  - app.rag.ingestion
  - app.rag.indexer
  - app.rag.search
  - app.rag.generate

If the solution introduces unnecessary alternatives such as LangChain, pdfplumber,
or unrelated frameworks when the existing stack already solves the problem,
flag that as a revision requirement.

Return your review in exactly this structure:

DECISION: APPROVE
or
DECISION: REVISE

ISSUES:
- List concrete issues.
- If there are no meaningful issues, write "None."

REVISION_INSTRUCTIONS:
- Give specific instructions the Coder Agent should follow.
- If approved, write "None."

User task:
{user_task}

Research findings:
{research_findings}

Coder solution:
{proposed_solution}
"""  # Gives the reviewer explicit criteria and the actual project stack.

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )  # Sends the solution to our local Reviewer Agent.

    review = response["message"]["content"]  # Extract the review text.

    decision = (
        "APPROVE"
        if "DECISION: APPROVE" in review.upper()
        else "REVISE"
    )  # Convert the review into a simple machine-readable decision.

    return {
        "decision": decision,
        "review": review,
    }  # Return both the structured decision and full reviewer feedback.


if __name__ == "__main__":
    test_task = (
        "Explain how page-aware ingestion should be used "
        "to support citations in a RAG application."
    )  # Same task used for our Coder test.

    test_research = """
Page-aware ingestion preserves the original PDF page number for source citations.
[Source: citation_test.pdf, page 1]
"""  # Research evidence available to the Coder.

    test_solution = """
Use pdfplumber to extract PDF pages and LangChain Chroma with
HuggingFaceEmbeddings to build the RAG pipeline.
"""  # Intentionally inconsistent solution so we can confirm the Reviewer catches it.

    result = review_solution(
        test_task,
        test_research,
        test_solution,
    )

    print("\nREVIEW DECISION:")
    print(result["decision"])

    print("\nREVIEW:")
    print(result["review"])