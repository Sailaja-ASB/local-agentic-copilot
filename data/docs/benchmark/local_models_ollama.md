# Running Local Models with Ollama

Ollama runs open-weight language models such as Qwen or Llama on a developer's own machine and exposes them through a local HTTP API on port 11434. Running locally keeps data private, works offline and has no per-token cost.

Models are usually quantised, for example to 4-bit weights, so a 3-billion-parameter model fits in a few gigabytes of memory. By default Ollama may process one request at a time per model; the OLLAMA_NUM_PARALLEL setting allows concurrent requests at the cost of more memory and slower individual generations.
