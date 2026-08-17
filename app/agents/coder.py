import ollama  # Lets the Coder Agent use our local Qwen model.


MODEL_NAME = "qwen3:4b"  # Local model used by the Coder Agent.


def create_solution(user_task: str, research_findings: str) -> dict:
    """Create a technical solution using the researcher's findings."""

    prompt = f"""
You are the Coder Agent in a technical AI system.

Your job is to create a clear technical solution based on the user's task
and the research findings provided by the Researcher Agent.

Rules:
1. Use the research findings as the primary source of truth.
2. Do not invent requirements that are not supported.
3. If code is needed, produce clean and readable code.
4. Explain important implementation decisions briefly.
5. Preserve relevant source citations from the research findings.
6. Do not claim the solution is verified yet; a Reviewer Agent will check it next.

User task:
{user_task}

Research findings:
{research_findings}
"""  # Gives Qwen a dedicated coding/solution-building role.

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )  # Sends the task and researcher findings to the local Coder Agent.

    solution = response["message"]["content"]  # Extract the proposed technical solution.

    return {
        "user_task": user_task,
        "research_findings": research_findings,
        "solution": solution,
    }  # Return structured data so the Reviewer Agent can inspect it later.


if __name__ == "__main__":
    test_task = (
        "Explain how page-aware ingestion should be used "
        "to support citations in a RAG application."
    )  # Temporary test task for the Coder Agent.

    test_research = """
Page-aware ingestion preserves the original PDF page number for source citations.
[Source: citation_test.pdf, page 1]
"""  # Temporary researcher findings used only to verify the Coder Agent.

    result = create_solution(
        test_task,
        test_research,
    )

    print("\nCODER TASK:")
    print(result["user_task"])

    print("\nCODER SOLUTION:")
    print(result["solution"])