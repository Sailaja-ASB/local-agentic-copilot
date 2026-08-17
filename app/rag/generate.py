import ollama  # Lets our Python app call the local Qwen model through Ollama.

from app.rag.search import search_documents  # Retrieves relevant evidence.


MODEL_NAME = "qwen3:4b"  # Local LLM installed in Ollama.

TOP_K = 3  # Number of retrieved chunks given to Qwen.


def build_context(matches: list[dict]) -> str:
    """Format retrieved chunks into grounded context with source citations."""

    context_parts = []

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
        )

    return "\n\n".join(context_parts)


def answer_question(question: str) -> dict:
    """Retrieve evidence and generate a grounded local answer."""

    # Step 1: Retrieve relevant document chunks.
    matches = search_documents(
        question,
        top_k=TOP_K,
    )

    # Step 2: Build citation-aware context.
    context = build_context(matches)

    # Step 3: Create the grounded prompt.
    prompt = f"""
You are a grounded technical assistant.

Answer the question using only the provided context.

Rules:
1. Do not use outside knowledge if the context does not support it.
2. If the answer is not present in the context, say:
   "I do not have enough information in the provided documents."
3. Keep the answer concise and technically accurate.
4. Cite the supporting source in the answer.
5. Use the exact source name and page number shown in the context.

Context:
{context}

Question:
{question}
"""

    # Step 4: Send the question and retrieved context to local Qwen.
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    # Step 5: Extract Qwen's generated answer.
    answer = response["message"]["content"]

    # Step 6: Keep only sources that Qwen actually cited.
    sources = []

    for match in matches:
        source = match["source"]
        page = match["page"]

        source_is_cited = source.lower() in answer.lower()

        page_is_cited = (
            page is None
            or f"page {page}" in answer.lower()
        )

        if source_is_cited and page_is_cited:
            source_record = {
                "source": source,
                "page": page,
            }

            # Avoid duplicate source entries.
            if source_record not in sources:
                sources.append(source_record)

    # IMPORTANT:
    # Return the structured result from the function.
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    # Test PDF-grounded retrieval and citation.
    question = "What does page-aware ingestion preserve?"

    result = answer_question(question)

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:
        if source["page"] is not None:
            print(
                f"- {source['source']} — page {source['page']}"
            )
        else:
            print(
                f"- {source['source']}"
            )