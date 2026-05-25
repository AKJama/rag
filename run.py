"""
One-shot setup: download the FiQA dataset and build both retrieval indexes.
Run this once before using any of the notebooks or evaluate.py.

    uv run python run.py
"""

from datasets import load_dataset

from config import FIQA_DIR
from src.retrievers import BM25Retriever, DenseRetriever


def download_data():
    if all((FIQA_DIR / f).exists() for f in ("corpus.parquet", "queries.parquet", "qrels.parquet")):
        print("Data already downloaded, skipping.")
        return

    print("Downloading FiQA-2018 from HuggingFace...")
    FIQA_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    queries = load_dataset("BeIR/fiqa", "queries", split="queries")
    qrels = load_dataset("BeIR/fiqa-qrels", split="test")

    corpus.to_parquet(FIQA_DIR / "corpus.parquet")
    queries.to_parquet(FIQA_DIR / "queries.parquet")
    qrels.to_parquet(FIQA_DIR / "qrels.parquet")

    print(f"  Corpus:  {len(corpus):>6} docs")
    print(f"  Queries: {len(queries):>6} queries")
    print(f"  Qrels:   {len(qrels):>6} judgments")


def build_indexes():
    print("\nBuilding BM25 index...")
    BM25Retriever()

    print("\nBuilding dense index (calls OpenAI embeddings API)...")
    DenseRetriever()


if __name__ == "__main__":
    download_data()
    build_indexes()
    print("\nDone. Run `uv run python notebooks/evaluate.py` to evaluate.")
