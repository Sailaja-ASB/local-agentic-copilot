import ollama  # Lets our Python app call the local Qwen model through Ollama.

from app.rag.search import search_documents  # Reuses our modular search layer to retrieve relevant evidence.


MODEL_NAME = "qwen3:4b"  # Local LLM already installed in Ollama.

TOP_K = 3  # Number of retrieved chunks we will give to Qwen as context.


def build_context(matches: list[dict]) -> str:
    """Format retrieved chunks into grounded context with source citations."""

    context_parts = []  # Stores each formatted evidence block.

    for match in matches:
        source = match["source"]  # Source filename for this chunk.

        page = match["page"]  # PDF page number when available.

        citation = (
            f"{source}, page {page}"
            if page is not None
            else source
        )  # Build a readable citation label.

        context_parts.append(
            f"[Source: {citation}]\n{match['text']}"
        )  # Attach source metadata directly to the evidence.

    return "\n\n".join(context_parts)  # Combine all retrieved evidence into one context string.


def answer_question(question: str) -> dict:
    """Retrieve evidence and generate a grounded local answer."""

    matches = search_documents(
        question,
        top_k=TOP_K,
    )  # Retrieve the most relevant document chunks.

    context = build_context(matches)  # Turn retrieved chunks into citation-aware context.

    prompt = f"""
You are a grounded technical assistant.

Answer the question using only the provided context.

Rules:
1. Do not use outside knowledge if the context does not support it.
2. If the answer is not present in the context, say:
   "I do not have enough information in the provided documents."
3. Keep the answer concise and technically accurate.
4. Cite the supporting source in the answer.

Context:
{context}

Question:
{question}
"""  # Creates strict grounding instructions to reduce unsupported answers.

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )  # Sends the retrieved evidence and question to local Qwen.

    answer = response["message"]["content"]  # Extract the generated answer text.

    sources = [
        {
            "source": match["source"],
            "page": match["page"],
        }
        for match in matches
    ]  # Keep structured citation metadata for future API/UI use.

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }  # Return structured output instead of only printing text.


if __name__ == "__main__":
    question = "What does RAG combine?"  # Temporary test question.

    result = answer_question(question)  # Run retrieval + grounded Qwen generation.

    print("\nQUESTION:")
    print(result["question"])  # Show the user's question.

    print("\nANSWER:")
    print(result["answer"])  # Show Qwen's grounded response.

    print("\nSOURCES:")

    for source in result["sources"]:
        if source["page"] is not None:
            print(f"- {source['source']} — page {source['page']}")  # PDF citation.
        else:
            print(f"- {source['source']}")  # TXT/Markdown citation.