# RAG Retrieval Pipeline

A retrieval pipeline built on the [FiQA-2018](https://sites.google.com/view/fiqa) financial QA benchmark, exploring how each stage of a modern retrieval stack improves result quality (measured by NDCG@10).

## Pipeline

| Stage | Method | NDCG@10 |
|---|---|---|
| 1 | BM25 (sparse) | 28.01 |
| 2 | Dense embeddings (`text-embedding-3-small`) | 40.06 |
| 3 | Hybrid via Reciprocal Rank Fusion | 33.72 |
| 4 | Hybrid + Cohere cross-encoder rerank | 47.71 |

## Stack

- **BM25** — `bm25s` for sparse retrieval
- **Dense** — OpenAI `text-embedding-3-small`, cosine similarity over a pre-built numpy matrix
- **Fusion** — Reciprocal Rank Fusion (k=60) to combine ranked lists without score normalisation
- **Rerank** — Cohere `rerank-v4.0-fast` cross-encoder over the top-50 RRF candidates

## Setup

```bash
uv sync
cp .env.example .env  # add OPENAI_API_KEY and COHERE_API_KEY
uv run python run.py  # download dataset and build all indexes (~$0.22 for embeddings)
```

Then to reproduce the evaluation:

```bash
uv run python notebooks/evaluate.py
```
