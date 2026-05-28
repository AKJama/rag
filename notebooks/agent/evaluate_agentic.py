"""
Ablation harness for Agentic RAG on the FiQA-2018 test set.

Compares toolset configurations (which search tools the agent may call) so we can
see whether each tool earns its place, on the same metric as notebooks/evaluate.py
(NDCG@10 x100) and the same fixed sample (SEED = 42).

We score the agent's own `retrieved_ids`: the ranked, pruned list it returns after
deciding which chunks are relevant. That agent-as-reranker step is the point, can it
beat the ~47% RAG best case, and which tools are pulling their weight?

Token discipline: agentic loops resend the full message history every turn, so cost
scales with snippet size x k x loop length. The caps live in agentic.py; here we keep
the sample small and run several configs. Every run is logged to logs/ablation_*.jsonl
so you can see exactly which tools the agent called and what it cost, after the fact.

Run from the repo root:
    uv run -m notebooks.agent.evaluate_agentic            # default 10 queries / config
    uv run -m notebooks.agent.evaluate_agentic --n 1      # 1-query sanity check
    uv run -m notebooks.agent.evaluate_agentic --n 50     # full eval
    uv run -m notebooks.agent.evaluate_agentic --configs hybrid grep+hybrid
"""

import argparse
import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic_ai import UsageLimits
from pydantic_ai.messages import ToolCallPart
from tqdm.asyncio import tqdm as tqdm_async

from notebooks.agent.agentic import AGENT_REQUEST_LIMIT, SEARCH_TOOLS, RunState, build_agent
from src.retrievers import FIQA_DIR

DEFAULT_SAMPLE_SIZE = 10  # small by default; agentic runs are ~3-8s each, costs add up fast
SEED = 42
MODEL = "openai:gpt-4.1-nano"  # cheap model while we shake out the eval framework
RAG_BEST_CASE = 47.0  # Hybrid + Rerank best case to beat

# Toolset ablations: does each tool earn its place?
CONFIGS: dict[str, list[str]] = {
    "grep_only": ["grep_pd"],
    "grep+bm25+dense": ["grep_pd", "bm25_search", "dense_search"],
    "hybrid": ["hybrid_search"],
    "grep+hybrid": ["grep_pd", "hybrid_search"],
}

LOG_DIR = Path(__file__).parent / "logs"

# CLI: --n sample size, --configs subset of CONFIGS
parser = argparse.ArgumentParser(description="Agentic RAG ablation harness.")
parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE, help="Queries per config (default 10).")
parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS), help="Configs to run.")
parser.add_argument("--concurrency", type=int, default=5, help="Parallel agent runs per config (default 5).")
args = parser.parse_args()
SAMPLE_SIZE = args.n
CONCURRENCY = args.concurrency
active_configs = {name: CONFIGS[name] for name in args.configs}
print(
    f"Plan: {len(active_configs)} configs x {SAMPLE_SIZE} queries = {len(active_configs) * SAMPLE_SIZE} runs "
    f"(concurrency={CONCURRENCY})."
)


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
# Step 2: Load queries + ground truth, pick the sample
# --------------------------------------------------------------

queries = pd.read_parquet(FIQA_DIR / "queries.parquet")
qrels_df = pd.read_parquet(FIQA_DIR / "qrels.parquet")

qrels: dict[str, dict[str, int]] = {
    str(qid): dict(zip(group["corpus-id"].astype(str), group["score"])) for qid, group in qrels_df.groupby("query-id")
}

queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()
sample = queries_with_qrels.sample(n=SAMPLE_SIZE, random_state=SEED)
print(f"Sampled {len(sample)} queries (seed={SEED}) from {len(queries_with_qrels)} qrel'd queries.")


# --------------------------------------------------------------
# Step 3: Per-run logging helper
# --------------------------------------------------------------

TOOL_NAMES = set(SEARCH_TOOLS) | {"read_doc"}


def tools_used(messages) -> list[str]:
    """Ordered list of tool calls the agent made, parsed from the run's messages."""
    used: list[str] = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart) and part.tool_name in TOOL_NAMES:
                used.append(part.tool_name)
    return used


# --------------------------------------------------------------
# Step 4: Run each toolset config over the sample, logging every run
# --------------------------------------------------------------

LOG_DIR.mkdir(exist_ok=True)
run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
log_path = LOG_DIR / f"ablation_{run_ts}.jsonl"
limits = UsageLimits(request_limit=AGENT_REQUEST_LIMIT)


async def run_one(config_name: str, config_agent, row, logf, lock: asyncio.Lock) -> dict:
    """Run the agent on one query, score it, and append a JSONL record (flushed live)."""
    query_id = str(row["_id"])
    relevant = qrels[query_id]
    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "config": config_name,
        "query_id": query_id,
    }
    try:
        result = await config_agent.run(row["text"], deps=RunState(), usage_limits=limits)
        usage = result.usage
        predicted = [str(d) for d in result.output.retrieved_ids]
        used = tools_used(result.all_messages())
        record |= {
            "ndcg": round(ndcg_at_k(predicted, relevant), 4),
            "tools_used": used,
            "retrieved_ids": predicted,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "requests": usage.requests,
        }
    except Exception as e:  # noqa: BLE001 - one bad query shouldn't kill the run
        record |= {"ndcg": 0.0, "error": f"{type(e).__name__}: {e}"}
        tqdm_async.write(f"  [{config_name}] query {query_id} failed: {record['error']}")

    # Flush after each line so the log streams in real time and you can
    # `Get-Content -Wait` it from another terminal while the run is in progress.
    async with lock:
        logf.write(json.dumps(record) + "\n")
        logf.flush()
    return record


async def run_config(config_name: str, tool_names: list[str], logf, lock: asyncio.Lock) -> dict:
    """Run all sampled queries for one config in parallel (capped by CONCURRENCY)."""
    config_agent = build_agent(tool_names, model=MODEL)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(row):
        async with sem:
            return await run_one(config_name, config_agent, row, logf, lock)

    t0 = time.perf_counter()
    tasks = [bounded(row) for _, row in sample.iterrows()]
    records = await tqdm_async.gather(*tasks, desc=config_name)
    elapsed = time.perf_counter() - t0

    n = len(records)
    tokens = sum((r.get("input_tokens") or 0) + (r.get("output_tokens") or 0) for r in records)
    return {
        "ndcg": float(np.mean([r["ndcg"] for r in records]) * 100),
        "tokens_per_query": tokens / n,
        "total_tokens": tokens,
        "requests_per_query": sum(r.get("requests", 0) for r in records) / n,
        "calls_per_query": sum(len(r.get("tools_used", [])) for r in records) / n,
        "failures": sum(1 for r in records if "error" in r),
        "seconds": elapsed,
    }


async def main() -> dict[str, dict]:
    summary: dict[str, dict] = {}
    lock = asyncio.Lock()
    with log_path.open("w", encoding="utf-8") as logf:
        # Run configs sequentially (parallel within each) to keep the comparison
        # table coming out in declared order and to keep per-config tqdm bars clean.
        for config_name, tool_names in active_configs.items():
            summary[config_name] = await run_config(config_name, tool_names, logf, lock)
    return summary


# --------------------------------------------------------------
# Step 5: Comparison table
# --------------------------------------------------------------

if __name__ == "__main__":
    summary = asyncio.run(main())

    print(f"\nAgentic RAG ablation - NDCG@10 x100 on FiQA ({len(sample)} queries, model={MODEL})")
    print(f"Per-run log: {log_path}")
    print("-" * 84)
    print(f"{'config':<20}{'NDCG@10':>9}{'tok/query':>12}{'total tok':>12}{'calls/q':>9}{'fails':>7}{'secs':>9}")
    print("-" * 84)
    for name, s in summary.items():
        print(
            f"{name:<20}{s['ndcg']:>9.2f}{s['tokens_per_query']:>12.0f}"
            f"{s['total_tokens']:>12}{s['calls_per_query']:>9.1f}{s['failures']:>7}{s['seconds']:>9.1f}"
        )
    print("-" * 84)
    print(f"{'RAG best case':<20}{RAG_BEST_CASE:>9.2f}")
    grand_total = sum(s["total_tokens"] for s in summary.values())
    print(f"\nTotal tokens across {len(active_configs)} configs: {grand_total}")
