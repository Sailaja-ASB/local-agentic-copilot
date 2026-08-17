import ollama  # Lets this agent call our local Qwen model.

from app.rag.search import search_documents  # Reuses our RAG retrieval layer.


MODEL_NAME = "qwen3:4b"  # Local model used by the Researcher Agent.

TOP_K = 3  # Number of evidence chunks the researcher will inspect.


def research(question: str) -> dict:
    """Retrieve evidence and summarize the key findings for downstream agents."""

    matches = search_documents(
        question,
        top_k=TOP_K,
    )  # Retrieve the most relevant chunks from our knowledge base.

    context_parts = []  # Stores evidence blocks with source metadata.

    for match in matches:
        source = match["source"]
        page = match["page"]

        citation = (
            f"{source}, page {page}"
            if page is not None
            else source
        )

        context_parts.append(
            f"[Source: {citation}]\n{match['text']}"
        )  # Attach source information directly to each retrieved chunk.

    context = "\n\n".join(context_parts)  # Combine all evidence into one researcher context.

    prompt = f"""
You are the Researcher Agent in a technical AI system.

Your job is to analyze retrieved evidence and produce concise technical findings
for another agent that will later create a solution.

Rules:
1. Use only the provided evidence.
2. Do not invent unsupported facts.
3. Identify the most important requirements, constraints, and technical details.
4. Preserve source citations.
5. If the evidence is insufficient, clearly say what information is missing.

Evidence:
{context}

User task:
{question}
"""  # Gives Qwen a focused research role instead of asking it for the final answer.

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )  # Run the local Researcher Agent.

    findings = response["message"]["content"]  # Extract the research summary.

    return {
        "question": question,
        "findings": findings,
        "matches": matches,
    }  # Return both the reasoning output and retrieved evidence for later agents.


if __name__ == "__main__":
    question = "What does page-aware ingestion preserve?"  # Temporary researcher test.

    result = research(question)

    print("\nRESEARCH QUESTION:")
    print(result["question"])

    print("\nRESEARCH FINDINGS:")
    print(result["findings"])