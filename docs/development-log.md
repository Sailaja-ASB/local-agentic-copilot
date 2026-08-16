# Development Log

## Milestone 1 — Local LLM Setup
- Installed Ollama on macOS.
- Downloaded and tested `qwen3:4b`.
- Confirmed local LLM inference works.

## Milestone 2 — Basic Retrieval
- Created an isolated Python virtual environment.
- Added Sentence Transformers and ChromaDB.
- Built document chunking and embedding generation.
- Stored embeddings in a persistent local Chroma database.
- Verified semantic retrieval from a test document.

## Milestone 3 — Local RAG
- Connected retrieved context to local Qwen through Ollama.
- Generated an answer grounded in retrieved evidence.

## Milestone 4 — Multi-Document Retrieval
- Added RAG, Docker, and FastAPI sample documents.
- Added source metadata to retrieved chunks.
- Verified that RAG-related questions ranked `rag.txt` first.

## Milestone 5 — Evaluation
- Created a fixed benchmark before further retrieval tuning.
- Added Top-1 Accuracy and Hit@3 metrics.
- Baseline result: 100% Top-1 and 100% Hit@3 on 6 sanity-check questions.