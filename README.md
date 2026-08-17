# Local Agentic Copilot

A local-first **Agentic RAG (Retrieval-Augmented Generation) system** that combines document ingestion, semantic and lexical retrieval, reranking, grounded generation, page-aware citations, deterministic guardrails, and a bounded multi-agent self-correction workflow.

The project is designed to run locally using **Ollama + Qwen**, reducing dependency on external LLM APIs while providing traceable, citation-aware responses over private documents.

---

## Key Features

### Local RAG Pipeline

- Local LLM inference using Ollama
- Qwen3 4B model for generation
- Sentence-Transformers embeddings
- ChromaDB vector storage
- Grounded generation using retrieved document context
- Source-aware answers

### Multi-Format Document Ingestion

Supports ingestion of:

- PDF
- Markdown
- TXT

The ingestion pipeline extracts document content, creates chunks, and preserves metadata required for retrieval and citations.

### Page-Aware PDF Citations

PDF ingestion preserves the original page number as metadata.

Example:

```text
QUESTION:
What does page-aware ingestion preserve?

ANSWER:
Page-aware ingestion preserves the original PDF page number for source citations.

SOURCES:
- citation_test.pdf — page 1
```

This allows generated answers to trace retrieved evidence back to the original PDF page.

### Hybrid Retrieval

The retrieval system combines:

- Dense semantic vector search
- BM25 lexical search

Semantic retrieval helps match concepts expressed using different wording, while BM25 improves retrieval for exact identifiers, technical terms, and keyword-heavy queries.

### Cross-Encoder Reranking

Retrieved candidates can be reranked using a CrossEncoder model.

Pipeline:

```text
Query
  ↓
Dense Retrieval + BM25
  ↓
Candidate Set
  ↓
Cross-Encoder Reranker
  ↓
Top Evidence
```

On the project's fixed 8-query advanced retrieval benchmark:

```text
Reranked Top-1 Accuracy: 8/8 = 100%
Reranked Hit@3:         8/8 = 100%
```

These results describe the included project benchmark and should not be interpreted as general RAG accuracy.

---

## Agentic Workflow

The project includes multiple specialized agents:

### Researcher Agent

Retrieves relevant evidence from the indexed knowledge base and produces grounded research findings.

### Coder Agent

Uses the research findings to produce a technical solution.

### Deterministic Guardrails

Before an LLM reviewer can approve a solution, deterministic checks validate important project constraints.

For example, the guardrail layer can reject solutions that introduce unsupported implementation choices or fail to preserve required citation metadata.

### Reviewer Agent

Performs semantic review after deterministic checks pass.

The Reviewer can return:

```text
APPROVE
```

or:

```text
REVISE
```

### Self-Correction Loop

Rejected solutions are sent back to the Coder with revision feedback.

```text
User Task
   ↓
Researcher
   ↓
Coder
   ↓
Deterministic Guardrails
   ↓
   ├── FAIL ─────→ REVISE
   │                 ↓
   │               Coder
   │                 ↓
   └──────────── Guardrails
                     ↓
                  PASS
                     ↓
               LLM Reviewer
                     ↓
              APPROVE / REVISE
```

The workflow uses a bounded revision limit to prevent infinite agent loops.

---

## Example Self-Correction

During validation, an initial solution introduced an unsupported PDF-processing implementation.

The deterministic guardrail rejected it:

```text
Iteration 0 — REVISE — GUARDRAIL
Guardrail passed: False
```

The Coder then produced a corrected solution using the project's expected stack.

The second iteration passed deterministic validation and reached the LLM Reviewer:

```text
Iteration 1 — APPROVE — LLM_REVIEWER
Guardrail passed: True
```

This demonstrates deterministic validation combined with LLM-based semantic review.

---

## Architecture

```text
                     ┌─────────────────────┐
                     │      User Task      │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │     Researcher      │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │       Coder         │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │    Guardrails       │
                     └──────────┬──────────┘
                                │
                       Pass     │     Fail
                     ┌──────────┴──────────┐
                     ▼                     ▼
              ┌─────────────┐        Coder Revision
              │  Reviewer   │              │
              └──────┬──────┘              │
                     │                     │
              Approve│Revise               │
                     │   └─────────────────┘
                     ▼
              ┌─────────────┐
              │Final Output │
              └─────────────┘
```

The agents are grounded by the underlying RAG system:

```text
Documents
   ↓
TXT / Markdown / PDF Ingestion
   ↓
Chunking + Metadata
   ↓
Sentence-Transformer Embeddings
   ↓
ChromaDB
   ↓
Dense + BM25 Retrieval
   ↓
Cross-Encoder Reranking
   ↓
Grounded Context
   ↓
Ollama / Qwen3
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Local LLM Runtime | Ollama |
| Local Model | Qwen3 4B |
| Embeddings | Sentence-Transformers |
| Vector Database | ChromaDB |
| Lexical Retrieval | BM25 |
| Reranking | CrossEncoder |
| PDF Processing | pypdf |
| Testing | pytest |
| Version Control | Git / GitHub |

---

## Project Structure

```text
local-agentic-copilot/
│
├── app/
│   ├── agents/
│   │   ├── researcher.py
│   │   ├── coder.py
│   │   ├── reviewer.py
│   │   ├── guardrails.py
│   │   └── orchestrator.py
│   │
│   └── rag/
│       ├── ingestion.py
│       ├── indexer.py
│       ├── retriever.py
│       ├── search.py
│       └── generate.py
│
├── data/
│   └── docs/
│
├── eval/
│   ├── evaluate_retrieval.py
│   ├── evaluate_advanced.py
│   ├── evaluate_hybrid.py
│   ├── evaluate_reranker.py
│   ├── evaluate_v1.py
│   ├── questions.json
│   └── advanced_questions.json
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_guardrails.py
│   └── test_orchestrator.py
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sailaja-ASB/local-agentic-copilot.git
cd local-agentic-copilot
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama separately and make sure it is running.

Pull the local model:

```bash
ollama pull qwen3:4b
```

---

## Running the RAG System

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Build the document index:

```bash
python -m app.rag.indexer
```

Run retrieval:

```bash
python -m app.rag.search
```

Run grounded generation:

```bash
python -m app.rag.generate
```

---

## Running the Agentic Workflow

```bash
python -m app.agents.orchestrator
```

The workflow executes:

```text
Researcher
   ↓
Coder
   ↓
Guardrails
   ↓
Reviewer
   ↓
Revision / Approval
```

---

## Testing

Run the complete automated test suite:

```bash
python -m pytest -s
```

Current V1 validation:

```text
5 passed
```

The tests cover:

- document ingestion
- page-aware PDF metadata
- deterministic guardrail rejection
- valid guardrail acceptance
- agent revision and approval behavior
- bounded revision behavior

---

## V1 Evaluation

Run:

```bash
python eval/evaluate_v1.py
```

The V1 evaluation runner validates:

```text
PASS | Full automated test suite
PASS | Advanced retrieval benchmark
PASS | Hybrid retrieval benchmark
PASS | Reranking benchmark

V1 checks passed: 4/4

V1 STATUS: READY
```

The fixed advanced reranking benchmark achieved:

```text
Top-1 Accuracy: 8/8 = 100%
Hit@3:          8/8 = 100%
```

Benchmark results are reported only for the included evaluation dataset and are not intended as claims of universal RAG accuracy.

---

## Engineering Goals

This project explores practical engineering patterns for local agentic AI systems:

- grounded generation instead of unrestricted LLM responses
- local-first inference
- hybrid information retrieval
- second-stage reranking
- verifiable document citations
- deterministic validation before LLM judgment
- bounded agent self-correction
- reproducible automated evaluation

---

## V1 Status

**READY**

The V1 core includes:

- document ingestion
- page-aware PDF metadata
- semantic retrieval
- BM25 retrieval
- hybrid retrieval
- cross-encoder reranking
- grounded local generation
- source citations
- Researcher/Coder/Reviewer agents
- deterministic guardrails
- bounded self-correction
- automated tests
- reproducible V1 evaluation

---

## Future Work

Potential V2 improvements include:

- FastAPI service layer
- interactive web UI
- document upload and indexing
- streaming responses
- conversation/session memory
- observability and tracing
- expanded evaluation datasets
- Docker packaging
- deployment