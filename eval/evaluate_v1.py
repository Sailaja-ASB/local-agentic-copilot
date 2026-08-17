import subprocess
import sys


def run_command(name: str, command: list[str]) -> bool:
    """Run one V1 validation command and report whether it passed."""

    print(f"\n{'=' * 60}")
    print(f"V1 CHECK: {name}")
    print("=" * 60)

    result = subprocess.run(
        command,
        text=True,
    )

    if result.returncode == 0:
        print(f"\nPASS: {name}")
        return True

    print(f"\nFAIL: {name}")
    return False


def main():
    """Run the complete automated V1 validation suite."""

    checks = [
        (
            "Full automated test suite",
            [sys.executable, "-m", "pytest", "-s"],
        ),
        (
            "Advanced retrieval benchmark",
            [sys.executable, "eval/evaluate_advanced.py"],
        ),
        (
            "Hybrid retrieval benchmark",
            [sys.executable, "eval/evaluate_hybrid.py"],
        ),
        (
            "Reranking benchmark",
            [sys.executable, "eval/evaluate_reranker.py"],
        ),
    ]

    results = []

    for name, command in checks:
        passed = run_command(
            name,
            command,
        )

        results.append(
            {
                "name": name,
                "passed": passed,
            }
        )

    print(f"\n{'=' * 60}")
    print("LOCAL AGENTIC COPILOT — V1 EVALUATION")
    print("=" * 60)

    passed_count = sum(
        result["passed"]
        for result in results
    )

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} | {result['name']}")

    print(
        f"\nV1 checks passed: "
        f"{passed_count}/{len(results)}"
    )

    if passed_count == len(results):
        print("\nV1 STATUS: READY")
        return 0

    print("\nV1 STATUS: NEEDS ATTENTION")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())