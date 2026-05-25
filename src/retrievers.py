"""
Polished versions of the BM25 and dense retrievers built under notebooks/ dir
"""

import os

import bm25s
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from config import DATA_DIR, INDEX_DIR

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


BM25_DIR = INDEX_DIR / "bm25"
DENSE_DIR = INDEX_DIR / "dense"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "corpus.parquet")


# --------------------------------------------------------------
# BM25
# --------------------------------------------------------------


class BM25Retriever:
    def __init__(self) -> None:
        if not BM25_DIR.exists():
            self.save_local()

        self._retriever = bm25s.BM25.load(str(BM25_DIR))
        self._doc_ids = (BM25_DIR / "doc_ids.txt").read_text().splitlines()

    def save_local(self, corpus=None):
        if corpus is None:
            corpus = load_corpus()

        doc_ids = corpus["_id"].tolist()
        doc_texts = corpus["text"].tolist()
        print(f"Indexing {len(doc_texts)} documents with BM25...")

        tokens = bm25s.tokenize(doc_texts, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(tokens)

        # Save the index plus the doc_ids in matching order.
        BM25_DIR.mkdir(parents=True, exist_ok=True)
        retriever.save(str(BM25_DIR))
        (BM25_DIR / "doc_ids.txt").write_text("\n".join(doc_ids))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([query], stopwords="en")
        indices, scores = self._retriever.retrieve(tokens, k=k)
        return [(self._doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())]


# --------------------------------------------------------------
# Dense
# --------------------------------------------------------------


class DenseRetriever:
    def __init__(self) -> None:
        self.corpus = load_corpus()
        self._doc_ids = self.corpus["_id"].tolist()

        self.embeddings_path = DENSE_DIR / "embeddings.npy"
        if not self.embeddings_path.exists():
            raw = self.save_local()
        else:
            raw = np.load(DENSE_DIR / "embeddings.npy")

        self._embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    # Index
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts and return a (len(texts), 1536) array."""
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return np.array([d.embedding for d in response.data], dtype=np.float32)

    def _build_index(self, doc_texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Embed the full corpus in batches with a progress bar."""
        chunks = []
        for i in tqdm(range(0, len(doc_texts), batch_size), desc="Embedding"):
            chunks.append(self._embed_batch(doc_texts[i : i + batch_size]))
        return np.vstack(chunks)  # stack batches into one (N, 1536) matrix

    # Save Index
    def save_local(self):
        # OpenAI rejects empty strings in the embeddings endpoint.
        doc_texts = [t.strip() or "[empty document]" for t in self.corpus["text"].tolist()]
        print(f"Embedding {len(doc_texts)} docs (~$0.22 at text-embedding-3-small)")

        doc_embeddings = self._build_index(doc_texts)
        np.save(self.embeddings_path, doc_embeddings)
        return doc_embeddings

    # Query
    def _embed_query(self, query: str) -> np.ndarray:
        vec = self._embed_batch([query])
        return vec.flatten() / np.linalg.norm(vec)

    # Search
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self._embeddings @ self._embed_query(query)
        # Get indices of top k elements
        top_k_indices = np.argpartition(-scores, k)[:k]
        # Sort those k by score (descending)
        top_k = top_k_indices[np.argsort(-scores[top_k_indices])]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
