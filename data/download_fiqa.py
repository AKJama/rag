"""
One-time fetch of the FiQA-2018 benchmark from the HuggingFace BeIR repos.

FiQA is a financial question answering benchmark: real finance questions
('Where should I park my rainy-day fund?') retrieved against forum posts and
opinion articles. 57,638 corpus docs, 648 test queries, ~2.6 relevant docs
per query. It is the closest stand-in BEIR has for a business knowledge base.

More info: https://sites.google.com/view/fiqa
"""

from pathlib import Path

from datasets import load_dataset

FIQA_DIR = Path(__file__).parent / "fiqa"
FIQA_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------
# Step 1: Pull the three pieces of a BEIR dataset
# --------------------------------------------------------------

# Every BEIR benchmark ships in the same shape:
#   - corpus: the documents to search over
#   - queries: the user queries
#   - qrels: the ground-truth (query_id, doc_id, relevance) triples

corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus")
queries = load_dataset("BeIR/fiqa", "queries", split="queries")
qrels = load_dataset("BeIR/fiqa-qrels", split="test")


# --------------------------------------------------------------
# Step 2: Cache as parquet so the other files load instantly
# --------------------------------------------------------------

corpus.to_parquet(FIQA_DIR / "corpus.parquet")
queries.to_parquet(FIQA_DIR / "queries.parquet")
qrels.to_parquet(FIQA_DIR / "qrels.parquet")


if __name__ == "__main__":
    print(f"Corpus:  {len(corpus):>6} docs    -> {FIQA_DIR / 'corpus.parquet'}")
    print(f"Queries: {len(queries):>6} queries -> {FIQA_DIR / 'queries.parquet'}")
    print(f"Qrels:   {len(qrels):>6} judgments -> {FIQA_DIR / 'qrels.parquet'}")
