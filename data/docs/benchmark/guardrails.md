# Guardrails for LLM Applications

Guardrails are checks placed before and after the model. Input guardrails validate length, reject empty or oversized requests, and screen for injection attempts or disallowed topics. Output guardrails check the answer before it reaches the user: they can block leaked secrets, remove personal data, and verify that every citation points to a chunk that was actually retrieved.

Citation verification catches a common failure where the model invents a source name that looks plausible. Guardrails should fail closed for safety checks and log every block so false positives can be reviewed.
