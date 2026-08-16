# System Architecture

## Current Goal
Build a local-first AI engineering assistant that can retrieve relevant evidence from documents and generate grounded answers using a locally hosted LLM.

## Current Architecture

User Question
    ↓
Embedding Model
    ↓
ChromaDB Semantic Retrieval
    ↓
Top Relevant Document Chunks
    ↓
Ollama
    ↓
Qwen3:4B
    ↓
Grounded Answer

## Components

### 1. Ollama
Runs the language model locally on the Mac.

Current model:
- `qwen3:4b`

Why:
- Local inference
- No cloud API required for the base system
- Suitable starting model for limited local hardware

### 2. Sentence Transformers
Embedding model:
- `sentence-transformers/all-MiniLM-L6-v2`

Purpose:
- Convert document chunks into numerical vectors
- Convert user questions into the same vector space
- Enable semantic similarity search

### 3. ChromaDB
Purpose:
- Persist document embeddings locally
- Store source metadata
- Retrieve the most semantically relevant chunks

### 4. Document Ingestion
Current supported format:
- TXT

Current process:
1. Read documents from `data/docs`
2. Split documents into chunks
3. Generate embeddings
4. Store chunks, embeddings, and source metadata in ChromaDB

### 5. Retrieval Evaluation
Current metrics:
- Top-1 Accuracy
- Hit@3

Current baseline:
- Top-1 Accuracy: 100%
- Hit@3: 100%

Important:
This baseline uses a very small sanity-check dataset and is not considered production-level accuracy.

## Planned Architecture

User
    ↓
Query Processing
    ↓
Hybrid Retrieval
    ├── Dense Semantic Search
    └── Keyword/BM25 Search
    ↓
Reranker
    ↓
Grounded Context
    ↓
Researcher Agent
    ↓
Coder Agent
    ↓
Reviewer Agent
    ↓
Reject → Coder Revises
    ↓
Approve
    ↓
Final Answer + Sources

## Planned Improvements
- PDF and Markdown ingestion
- Better chunking strategies
- Hybrid retrieval
- Reranking
- Larger evaluation benchmark
- Grounded citations
- Researcher/Coder/Reviewer workflow
- Reviewer feedback loop
- Model routing
- FastAPI backend
- Tests and structured logging
- Docker
- UI
- Deployment