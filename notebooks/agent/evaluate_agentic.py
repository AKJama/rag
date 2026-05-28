"""
NDCG@10 for the Agentic RAG approach on the FiQA-2018 test set.

Mirrors notebooks/evaluate.py: same metric (NDCG@10 x100), same fixed sample
(RERANK_SAMPLE_SIZE = 50, SEED = 42) so the agentic numbers line up with the
RAG pipeline numbers for an apples-to-apples comparison.

Where the RAG pipeline scores a retriever's top-10, here we score the agent's
own `retrieved_ids`: the ranked, pruned list the agent returns after deciding
which chunks (from bm25_search / dense_search / grep_pd) are actually relevant.
That agent-as-reranker step is the whole point, can it beat the ~47% RAG best case?

Agentic runs are slow (5-15s each) and cost 3-10x the tokens of vanilla RAG, so
this evaluates serially with a tqdm bar and a per-query request cap.

Run from the repo root:
    uv run -m notebooks.agent.evaluate_agentic
"""

import math
import time

import numpy as np
import pandas as pd
from pydantic_ai import UsageLimits
from tqdm import tqdm

from notebooks.agent.agentic import AGENT_REQUEST_LIMIT, agent
from src.retrievers import FIQA_DIR

RERANK_SAMPLE_SIZE = 50
SEED = 42
RAG_BEST_CASE = 47.0  # Hybrid + Rerank best case to beat


# --------------------------------------------------------------
# Step 1: NDCG@k (same definition as notebooks/evaluate.py)
# --------------------------------------------------------------


def ndcg_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Normalized discounted cumulative gain for a single query."""
    dcg = sum(relevant.get(doc_id, 0) / math.log2(rank + 2) for rank, doc_id in enumerate(predicted_ids[:k]))
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------
# Step 2: Load queries + ground truth, pick the same sample as evaluate.py
# --------------------------------------------------------------

queries = pd.read_parquet(FIQA_DIR / "queries.parquet")
qrels_df = pd.read_parquet(FIQA_DIR / "qrels.parquet")

qrels: dict[str, dict[str, int]] = {
    str(qid): dict(zip(group["corpus-id"].astype(str), group["score"])) for qid, group in qrels_df.groupby("query-id")
}

queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()
sample = queries_with_qrels.sample(n=RERANK_SAMPLE_SIZE, random_state=SEED)
print(f"Evaluating Agentic RAG on {len(sample)} queries (sampled from {len(queries_with_qrels)})")


# --------------------------------------------------------------
# Step 3: Run the agent per query, score its retrieved_ids
# --------------------------------------------------------------


def agentic_topk(query: str) -> list[str]:
    """Run the agent and return its ranked retrieved_ids (its final top-k)."""
    result = agent.run_sync(query, usage_limits=UsageLimits(request_limit=AGENT_REQUEST_LIMIT))
    usage = result.usage
    return [str(d) for d in result.output.retrieved_ids], usage


scores: list[float] = []
failures = 0
total_tokens = 0
total_requests = 0
start = time.perf_counter()

for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Agentic"):
    query_id = str(row["_id"])
    query_text = row["text"]
    relevant = qrels[query_id]

    try:
        predicted_ids, usage = agentic_topk(query_text)
        total_tokens += (usage.input_tokens or 0) + (usage.output_tokens or 0)
        total_requests += usage.requests
        scores.append(ndcg_at_k(predicted_ids, relevant))
    except Exception as e:  # noqa: BLE001 - one bad query shouldn't kill the run
        failures += 1
        scores.append(0.0)
        tqdm.write(f"  query {query_id} failed: {type(e).__name__}: {e}")

elapsed = time.perf_counter() - start


# --------------------------------------------------------------
# Step 4: Print the result next to the RAG baseline
# --------------------------------------------------------------

if __name__ == "__main__":
    mean_ndcg = np.mean(scores) * 100
    print(f"\nNDCG@10 x100 on FiQA ({len(sample)} sampled test queries)")
    print("-" * 42)
    print(f"  {'Agentic RAG':<22} {mean_ndcg:.2f}")
    print(f"  {'RAG best case (target)':<22} {RAG_BEST_CASE:.2f}")
    verdict = "BEATS" if mean_ndcg > RAG_BEST_CASE else "below"
    print(f"\n  -> Agentic {verdict} the RAG best case ({mean_ndcg:.2f} vs {RAG_BEST_CASE:.2f})")
    print(
        f"\nCost: {total_requests} LLM requests, {total_tokens} tokens, "
        f"{failures} failures, {elapsed:.0f}s total ({elapsed / len(sample):.1f}s/query)"
    )
