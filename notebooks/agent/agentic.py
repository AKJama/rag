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
from dataclasses import dataclass

import nest_asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, UsageLimits

from src.fusion import hybrid_candidates
from src.retrievers import BM25Retriever, DenseRetriever, load_corpus

nest_asyncio.apply()

DEFAULT_MODEL = "openai:gpt-4.1-nano"

# Caps are kept deliberately small: an agentic loop resends the full message
# history (every prior tool result) on every turn, so large snippets, high k,
# and long loops inflate token cost super-linearly. Tune these to trade
# recall for cost.
MAX_SEARCH_K = 15
MAX_GREP_RESULTS = 12
DEFAULT_SEARCH_K = 8
SNIPPET_CHARS = 120
READ_MAX_CHARS = 2000
AGENT_REQUEST_LIMIT = 15
READ_DOC_CAP = 3  # hard cap on read_doc calls per agent run; snippets cover the rest

logger = logging.getLogger(__name__)


@dataclass
class RunState:
    """Per-run mutable state. A fresh instance is passed via `deps=` for every agent run.

    Used to enforce hard caps that the LLM cannot override (e.g. read_doc abuse).
    """

    read_doc_calls: int = 0
    read_doc_cap: int = READ_DOC_CAP


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


def grep_pd(pattern: str, max_results: int = DEFAULT_SEARCH_K) -> str:
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


def bm25_search(query: str, k: int = DEFAULT_SEARCH_K) -> str:
    """Sparse keyword (BM25) retrieval. Returns up to `k` `doc_id: snippet` lines, best first.

    `k` is clamped to MAX_SEARCH_K. Good for exact-term financial questions.
    """
    logger.info("bm25_search(query=%r, k=%d)", query, k)
    if k < 1:
        return "Error: k must be 1 or greater."
    results = _bm25.search(query, k=min(k, MAX_SEARCH_K))
    return _search_lines(results) or f"No results for query: {query}"


def dense_search(query: str, k: int = DEFAULT_SEARCH_K) -> str:
    """Semantic (embedding) retrieval. Returns up to `k` `doc_id: snippet` lines, best first.

    `k` is clamped to MAX_SEARCH_K. Good for paraphrase-heavy questions.
    """
    logger.info("dense_search(query=%r, k=%d)", query, k)
    if k < 1:
        return "Error: k must be 1 or greater."
    results = _dense.search(query, k=min(k, MAX_SEARCH_K))
    return _search_lines(results) or f"No results for query: {query}"


def hybrid_search(query: str, k: int = DEFAULT_SEARCH_K) -> str:
    """Hybrid retrieval: BM25 + dense fused with Reciprocal Rank Fusion (RRF).

    Returns up to `k` `doc_id: snippet` lines, best first. `k` is clamped to
    MAX_SEARCH_K. A strong single-shot default that blends keyword and semantic signal.
    """
    logger.info("hybrid_search(query=%r, k=%d)", query, k)
    if k < 1:
        return "Error: k must be 1 or greater."
    results = hybrid_candidates(query, _bm25, _dense, candidate_k=min(k, MAX_SEARCH_K))
    return _search_lines(results) or f"No results for query: {query}"


# --------------------------------------------------------------
# Step 4: read a single doc by _id
# --------------------------------------------------------------


def read_doc(ctx: RunContext[RunState], doc_id: str) -> str:
    """Return the full text of a single corpus document by its _id.

    Hard-capped at `RunState.read_doc_cap` calls per run; once exceeded the tool
    returns an error and the agent must judge from snippets already in context.
    """
    state = ctx.deps
    if state.read_doc_calls >= state.read_doc_cap:
        logger.info("read_doc(doc_id=%r) BLOCKED (cap %d reached)", doc_id, state.read_doc_cap)
        return (
            f"Error: read_doc cap of {state.read_doc_cap} reached for this query. "
            "Decide from the snippets already shown."
        )
    state.read_doc_calls += 1
    logger.info("read_doc(doc_id=%r) call %d/%d", doc_id, state.read_doc_calls, state.read_doc_cap)

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


# Registry of the available search tools, keyed by name. The eval harness picks
# subsets of these to ablate which tools earn their place. `read_doc` is always
# attached on top, since reading a candidate is orthogonal to how it was found.
SEARCH_TOOLS = {
    "grep_pd": grep_pd,
    "bm25_search": bm25_search,
    "dense_search": dense_search,
    "hybrid_search": hybrid_search,
}

_INSTRUCTIONS = (
    "You retrieve from a corpus of financial Q&A passages to answer the query. "
    "Your available search tools are: {available}. Use them to gather candidate doc_ids, "
    "then use read_doc sparingly (hard cap: {read_cap} calls per query) to verify the "
    "most promising candidates. Snippets already give you the gist; only read full text "
    "when you genuinely cannot rank from the snippet. "
    "Return retrieved_ids: up to 10 doc_ids ranked most-relevant-first to the query. "
    "Reordering and inclusion are your job, drop irrelevant candidates and promote the best. "
    "You may include any doc_id found via any tool. Be economical with tool calls. "
    "Back the answer with citations."
)


def build_agent(tool_names: list[str], model: str = DEFAULT_MODEL) -> Agent[RunState, SearchAnswer]:
    """Build an Agent exposing the named search tools (plus read_doc).

    `tool_names` is any subset of SEARCH_TOOLS keys; used by the ablation harness.
    The agent expects a fresh `RunState` instance passed via `deps=` on every run.
    """
    tools = [SEARCH_TOOLS[name] for name in tool_names] + [read_doc]
    return Agent(
        model,
        deps_type=RunState,
        tools=tools,
        output_type=SearchAnswer,
        instructions=_INSTRUCTIONS.format(available=", ".join(tool_names), read_cap=READ_DOC_CAP),
    )


# Default agent for the demo below: every search tool available.
agent = build_agent(list(SEARCH_TOOLS))


# --------------------------------------------------------------
# Step 7: Run it with a turn cap
# --------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start = time.perf_counter()
    result = agent.run_sync(
        "What is the difference between a Roth IRA and a traditional IRA?",
        deps=RunState(),
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
