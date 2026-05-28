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

## Agentic RAG Ablation

An agentic RAG system using `pydantic-ai` that lets an LLM choose which retrieval tools to call and how to rank results. Preliminary ablation (N=30 queries, `gpt-4.1-nano`):

| Config | NDCG@10 | Tokens/Query | Calls/Query |
|---|---|---|---|
| `grep_only` | 9.53 | 4,310 | 4.4 |
| `grep+bm25+dense` | 13.38 | 3,585 | 2.7 |
| `hybrid` | **24.65** | 2,310 | 2.0 |
| `grep+hybrid` | 23.04 | 2,721 | 2.3 |
| RAG best case (hybrid + rerank) | 47.00 | — | — |

**Key findings:**
- `hybrid_search` (pre-fused BM25+dense via RRF) is the most effective single tool
- Adding `grep_pd` to hybrid provides no benefit and increases cost
- Individual retrievers (`grep+bm25+dense`) underperform the pre-fused hybrid
- The agentic approach reaches ~52% of the RAG best case

Run the ablation:

```bash
uv run -m notebooks.agent.evaluate_agentic --n 30 --concurrency 5
```
