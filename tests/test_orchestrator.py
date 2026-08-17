from unittest.mock import patch  # Lets us replace real agents with fast fake responses during tests.

from app.agents.orchestrator import run_agentic_workflow  # Imports the workflow we want to verify.


def test_orchestrator_approves_after_revision():
    """Workflow should revise a rejected solution and stop once the reviewer approves it."""

    fake_research = {
        "findings": "Use pypdf, chromadb, and preserve page metadata.",
        "matches": [],
    }  # Pretends the Researcher already returned grounded findings.

    first_coder_output = {
        "solution": "Use pdfplumber for PDF processing."
    }  # Intentionally bad first solution so guardrails should reject it.

    revised_coder_output = {
        "solution": (
            "Use pypdf PdfReader for PDF processing. "
            "Preserve page metadata and store it with chunks in chromadb."
        )
    }  # Corrected solution that should pass guardrails.

    approved_review = {
        "decision": "APPROVE",
        "review": (
            "DECISION: APPROVE\n\n"
            "ISSUES:\n- None.\n\n"
            "REVISION_INSTRUCTIONS:\n- None."
        ),
    }  # Fake LLM Reviewer approval so the test does not call Qwen.

    with patch(
        "app.agents.orchestrator.research",
        return_value=fake_research,
    ), patch(
        "app.agents.orchestrator.create_solution",
        side_effect=[
            first_coder_output,
            revised_coder_output,
        ],
    ), patch(
        "app.agents.orchestrator.review_solution",
        return_value=approved_review,
    ):

        result = run_agentic_workflow(
            "Explain page-aware ingestion for citations."
        )  # Runs the real orchestration logic using fake agent responses.

    assert result["status"] == "APPROVED"  # Workflow should eventually succeed.

    assert len(result["review_history"]) == 2  # One rejection + one approval should be recorded.

    assert result["review_history"][0]["decision"] == "REVISE"  # First solution should fail guardrails.

    assert result["review_history"][0]["review_source"] == "GUARDRAIL"  # Confirms deterministic rejection happened first.

    assert result["review_history"][1]["decision"] == "APPROVE"  # Revised solution should be approved.

    assert result["review_history"][1]["review_source"] == "LLM_REVIEWER"  # Reviewer should only run after guardrails pass.


def test_orchestrator_stops_at_revision_limit():
    """Workflow should stop instead of looping forever when every solution keeps failing."""

    fake_research = {
        "findings": "Use pypdf, chromadb, and preserve page metadata.",
        "matches": [],
    }

    always_bad_solution = {
        "solution": "Use pdfplumber and LangChain."
    }  # Every generated solution intentionally violates guardrails.

    with patch(
        "app.agents.orchestrator.research",
        return_value=fake_research,
    ), patch(
        "app.agents.orchestrator.create_solution",
        return_value=always_bad_solution,
    ):

        result = run_agentic_workflow(
            "Explain page-aware ingestion for citations."
        )  # Exercises the bounded revision loop.

    assert result["status"] == "MAX_REVISIONS_REACHED"  # The workflow must terminate safely.

    assert len(result["review_history"]) == 3  # Initial attempt + two allowed revisions.

    assert all(
        review["decision"] == "REVISE"
        for review in result["review_history"]
    )  # Every attempt should be rejected.

    assert all(
        review["review_source"] == "GUARDRAIL"
        for review in result["review_history"]
    )  # Guardrails should prevent bad solutions from reaching the LLM Reviewer.