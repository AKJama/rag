"""
Agentic RAG over the FiQA-2018 parquet corpus.

The agent searches ~57k financial passages (rows in corpus.parquet, no files/markdown)
using a few simple tools, then returns its own ranked top-10 doc_ids. Tools:
  - grep_pd:      pandas regex over corpus text (exact keyword/symbol match)
  - bm25_search:  sparse keyword retrieval (src.retrievers.BM25Retriever)
  - dense_search: semantic retrieval (src.retrievers.DenseRetriever)
  - read_doc:     full text of a single doc by _id

Bounded outputs, tool-call logging, and agent-readable errors throughout.
"""

import logging
import re
import time

import nest_asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits

from src.retrievers import BM25Retriever, DenseRetriever, load_corpus

nest_asyncio.apply()

MAX_SEARCH_K = 50
MAX_GREP_RESULTS = 50
SNIPPET_CHARS = 200
READ_MAX_CHARS = 4000
AGENT_REQUEST_LIMIT = 20

logger = logging.getLogger(__name__)

# Loaded once at import; reused across every query/tool call.
_corpus = load_corpus()
_corpus_by_id = _corpus.set_index("_id")
_bm25 = BM25Retriever()
_dense = DenseRetriever()


# --------------------------------------------------------------
# Step 1: snippet helper
# --------------------------------------------------------------


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Collapse whitespace and truncate a passage for compact tool output."""
    one_line = " ".join(str(text).split())
    return one_line[:limit] + ("..." if len(one_line) > limit else "")


def _search_lines(results: list[tuple[str, float]]) -> str:
    return "\n".join(f"{doc_id}: {_snippet(_corpus_by_id.loc[doc_id, 'text'])}" for doc_id, _ in results)


# --------------------------------------------------------------
# Step 2: grep_pd (pandas regex over corpus text)
# --------------------------------------------------------------


def grep_pd(pattern: str, max_results: int = 30) -> str:
    """Regex search over corpus text (case-insensitive). Returns `doc_id: snippet` lines.

    Use for exact keywords, symbols, or phrases. `pattern` is a regular expression.
    Capped at MAX_GREP_RESULTS matches for safety.
    """
    logger.info("grep_pd(pattern=%r, max_results=%d)", pattern, max_results)

    if max_results < 1:
        return "Error: max_results must be 1 or greater."
    max_results = min(max_results, MAX_GREP_RESULTS)

    try:
        mask = _corpus["text"].str.contains(pattern, regex=True, case=False, na=False)
    except re.error as e:
        return f"Error: invalid regex {pattern!r}: {e}"

    total = int(mask.sum())
    hits = _corpus[mask].head(max_results)
    if hits.empty:
        return f"No matches found for pattern: {pattern}"

    lines = [f"{doc_id}: {_snippet(text)}" for doc_id, text in zip(hits["_id"], hits["text"])]
    if total > max_results:
        lines.append(f"... {total - max_results} more matches. Refine the pattern.")
    return "\n".join(lines)


# --------------------------------------------------------------
# Step 3: bm25 + dense search (imported retrievers)
# --------------------------------------------------------------


def bm25_search(query: str, k: int = 10) -> str:
    """Sparse keyword (BM25) retrieval. Returns up to `k` `doc_id: snippet` lines, best first.

    `k` is clamped to MAX_SEARCH_K. Good for exact-term financial questions.
    """
    logger.info("bm25_search(query=%r, k=%d)", query, k)
    if k < 1:
        return "Error: k must be 1 or greater."
    results = _bm25.search(query, k=min(k, MAX_SEARCH_K))
    return _search_lines(results) or f"No results for query: {query}"


def dense_search(query: str, k: int = 10) -> str:
    """Semantic (embedding) retrieval. Returns up to `k` `doc_id: snippet` lines, best first.

    `k` is clamped to MAX_SEARCH_K. Good for paraphrase-heavy questions.
    """
    logger.info("dense_search(query=%r, k=%d)", query, k)
    if k < 1:
        return "Error: k must be 1 or greater."
    results = _dense.search(query, k=min(k, MAX_SEARCH_K))
    return _search_lines(results) or f"No results for query: {query}"


# --------------------------------------------------------------
# Step 4: read a single doc by _id
# --------------------------------------------------------------


def read_doc(doc_id: str) -> str:
    """Return the full text of a single corpus document by its _id."""
    logger.info("read_doc(doc_id=%r)", doc_id)
    if doc_id not in _corpus_by_id.index:
        return f"Error: doc_id {doc_id!r} not found in corpus."
    text = str(_corpus_by_id.loc[doc_id, "text"])
    if len(text) > READ_MAX_CHARS:
        return text[:READ_MAX_CHARS] + f"\n... truncated ({len(text)} chars total)."
    return text


# --------------------------------------------------------------
# Step 5: Structured answer models
# --------------------------------------------------------------


class Citation(BaseModel):
    """One source backing a claim in the answer."""

    doc_id: str = Field(description="The corpus _id of the cited document, e.g. '566392'")
    quote: str = Field(description="Exact text from the document that supports the claim")


class SearchAnswer(BaseModel):
    """Structured answer plus the agent's own ranked retrieval list."""

    answer: str = Field(description="The answer in plain English")
    citations: list[Citation] = Field(description="Documents and quotes that support the answer")
    retrieved_ids: list[str] = Field(
        description=(
            "Up to 10 corpus _ids ranked most-relevant first. This is the agent's final "
            "retrieval ranking: reorder and drop candidates as you see fit."
        )
    )


# --------------------------------------------------------------
# Step 6: Production agent
# --------------------------------------------------------------


agent = Agent(
    # "openai:gpt-5.5",
    "openai:gpt-4.1-nano",  # faster
    tools=[grep_pd, bm25_search, dense_search, read_doc],
    output_type=SearchAnswer,
    instructions=(
        "You retrieve from a corpus of financial Q&A passages to answer the query. "
        "Use bm25_search and dense_search to gather candidate doc_ids, and grep_pd for exact "
        "keyword/symbol matches. Use read_doc to inspect full passages before judging relevance. "
        "Then return retrieved_ids: up to 10 doc_ids ranked most-relevant-first to the query. "
        "Reordering and inclusion are your job, drop irrelevant candidates and promote the best. "
        "You may include any doc_id found via any tool. Back the answer with citations."
    ),
)


# --------------------------------------------------------------
# Step 7: Run it with a turn cap
# --------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start = time.perf_counter()
    result = agent.run_sync(
        "What is the difference between a Roth IRA and a traditional IRA?",
        usage_limits=UsageLimits(request_limit=AGENT_REQUEST_LIMIT),
    )
    elapsed = time.perf_counter() - start

    print("\nAgent:", result.output.answer)
    print("\nRetrieved (ranked):", result.output.retrieved_ids)
    print("\nCitations:")
    for citation in result.output.citations:
        print(f"  - {citation.doc_id}")
        for line in citation.quote.splitlines():
            print(f"      {line}")

    usage = result.usage
    print(
        f"\nUsage: {usage.requests} requests, {usage.tool_calls} tool calls, "
        f"{usage.input_tokens} input + {usage.output_tokens} output tokens, "
        f"{elapsed:.1f}s"
    )
