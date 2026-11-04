"""Hybrid retrieval: ChromaDB dense search + BM25, fused with Reciprocal Rank Fusion.

Why fuse?  Dense embeddings capture meaning ("how does it retry failed
connections" ~ max_retries logic) but are fuzzy on exact names. BM25 nails
exact identifiers (`should_strip_auth`) but knows nothing about synonyms.
They fail on different queries, so combining them beats either alone.

Why RRF instead of adding scores?  Cosine similarities (~0.4-0.8) and BM25
scores (~0-30) live on incomparable scales. RRF ignores raw scores and only
uses each document's *rank* in each list:

    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

k (=60, the value from the original Cormack et al. 2009 paper) damps the
advantage of the very top ranks so that one retriever can't dominate; a doc
ranked moderately well by both beats a doc ranked #1 by one and absent from
the other.
"""

from collections import defaultdict
from typing import Literal

from langchain_core.documents import Document

from ingest.index import get_vectorstore
from retrieval.bm25 import BM25Index

Mode = Literal["hybrid", "dense", "bm25"]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, fetch_k: int = 30, rrf_k: int = 60):
        self.store = get_vectorstore()
        raw = self.store.get(include=["documents", "metadatas"])
        if not raw["ids"]:
            raise RuntimeError("Chroma index is empty - run `uv run python -m ingest.index` first.")
        self.docs = {
            i: Document(page_content=text, metadata=meta)
            for i, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        }
        self.bm25 = BM25Index(list(self.docs.values()))
        self.fetch_k = fetch_k  # candidates pulled from each retriever before fusion
        self.rrf_k = rrf_k

    def dense_ids(self, query: str, k: int) -> list[str]:
        return [d.metadata["id"] for d in self.store.similarity_search(query, k=k)]

    def bm25_ids(self, query: str, k: int) -> list[str]:
        return [i for i, _ in self.bm25.search(query, k=k)]

    def search_ids(self, query: str, k: int = 10, mode: Mode = "hybrid") -> list[str]:
        if mode == "dense":
            return self.dense_ids(query, k)
        if mode == "bm25":
            return self.bm25_ids(query, k)
        fused = reciprocal_rank_fusion(
            [self.dense_ids(query, self.fetch_k), self.bm25_ids(query, self.fetch_k)],
            k=self.rrf_k,
        )
        return [i for i, _ in fused[:k]]

    def search(self, query: str, k: int = 10, mode: Mode = "hybrid") -> list[Document]:
        return [self.docs[i] for i in self.search_ids(query, k, mode)]


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "how are redirects handled"
    r = HybridRetriever()
    for mode in ("dense", "bm25", "hybrid"):
        print(f"\n== {mode} ==")
        for i in r.search_ids(q, k=5, mode=mode):
            print("  ", i)
