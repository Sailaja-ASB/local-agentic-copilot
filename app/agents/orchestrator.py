from app.agents.researcher import research  # Runs grounded retrieval + research.
from app.agents.coder import create_solution  # Builds a technical solution from research findings.
from app.agents.reviewer import review_solution  # Runs semantic LLM review.
from app.agents.guardrails import validate_solution  # Runs deterministic stack checks first.


MAX_REVISIONS = 2  # Prevents endless Coder ↔ Reviewer loops.


def run_agentic_workflow(user_task: str) -> dict:
    """Run Researcher → Coder → Guardrails → Reviewer with bounded revisions."""

    # Step 1: Research the task using our RAG system.
    research_result = research(user_task)

    research_findings = research_result["findings"]


    # Step 2: Create the first proposed solution.
    coder_result = create_solution(
        user_task,
        research_findings,
    )

    current_solution = coder_result["solution"]


    # Step 3: Store every review/guardrail decision.
    review_history = []


    # Step 4: Run the approval/revision loop.
    for revision_number in range(MAX_REVISIONS + 1):

        # Run deterministic checks before the LLM Reviewer.
        guardrail_result = validate_solution(
            user_task,
            current_solution,
        )


        # If deterministic checks fail, force a revision.
        if not guardrail_result["passed"]:

            guardrail_feedback = "\n".join(
                f"- {violation}"
                for violation in guardrail_result["violations"]
            )

            review_result = {
                "decision": "REVISE",
                "review": (
                    "DECISION: REVISE\n\n"
                    "DETERMINISTIC GUARDRAIL VIOLATIONS:\n"
                    f"{guardrail_feedback}\n\n"
                    "REVISION_INSTRUCTIONS:\n"
                    "- Fix every deterministic guardrail violation.\n"
                    "- Stay consistent with the existing project stack.\n"
                    "- Do not introduce unsupported libraries."
                ),
            }

            review_source = "GUARDRAIL"


        # Only call the LLM Reviewer after deterministic checks pass.
        else:

            review_result = review_solution(
                user_task,
                research_findings,
                current_solution,
            )

            review_source = "LLM_REVIEWER"


        # Save this iteration for auditing/debugging.
        review_history.append(
            {
                "iteration": revision_number,
                "review_source": review_source,
                "guardrail_passed": guardrail_result["passed"],
                "guardrail_violations": guardrail_result["violations"],
                "decision": review_result["decision"],
                "review": review_result["review"],
            }
        )


        # If approved, finish immediately.
        if review_result["decision"] == "APPROVE":

            return {
                "status": "APPROVED",
                "user_task": user_task,
                "research_findings": research_findings,
                "final_solution": current_solution,
                "review_history": review_history,
            }


        # Stop when the maximum number of revisions has been reached.
        if revision_number == MAX_REVISIONS:
            break


        # Build revision instructions for the Coder.
        revision_prompt = f"""
The previous solution was rejected.

Reviewer/guardrail feedback:

{review_result["review"]}

Revise the previous solution.

Requirements:
1. Fix every issue listed above.
2. Stay consistent with the existing project stack.
3. Use pypdf for PDF processing.
4. Use sentence-transformers for embeddings.
5. Use chromadb as the vector database.
6. Use Ollama with qwen3:4b for local generation.
7. Preserve page metadata for PDF citations.
8. Do not introduce unnecessary libraries.
9. Keep relevant source citations.
10. Produce a technically correct revised solution.
11. In the revised final solution, do not discuss or list rejected alternative libraries.
    Simply implement the correct existing project stack.
"""

        # Send the review feedback back to the Coder.
        coder_result = create_solution(
            user_task,
            research_findings + "\n\n" + revision_prompt,
        )

        current_solution = coder_result["solution"]


    # If approval never happens, return the best available result.
    return {
        "status": "MAX_REVISIONS_REACHED",
        "user_task": user_task,
        "research_findings": research_findings,
        "final_solution": current_solution,
        "review_history": review_history,
    }


if __name__ == "__main__":

    task = (
        "Explain how page-aware ingestion should be used "
        "to support citations in a RAG application."
    )

    result = run_agentic_workflow(task)

    print("\nWORKFLOW STATUS:")
    print(result["status"])

    print("\nFINAL SOLUTION:")
    print(result["final_solution"])

    print("\nREVIEW HISTORY:")

    for review in result["review_history"]:

        print(
            f"\nIteration {review['iteration']} "
            f"— {review['decision']} "
            f"— {review['review_source']}"
        )

        print(
            f"Guardrail passed: {review['guardrail_passed']}"
        )

        if review["guardrail_violations"]:

            print("\nGuardrail violations:")

            for violation in review["guardrail_violations"]:
                print(f"- {violation}")

        print("\nReview:")
        print(review["review"])