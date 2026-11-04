"""In-memory BM25 (sparse / lexical) index over the code chunks."""

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from retrieval.tokenizer import tokenize


class BM25Index:
    def __init__(self, docs: list[Document]):
        self.ids = [d.metadata["id"] for d in docs]
        # Index page_content (context header + code), so file/class/function
        # names from the header are searchable too.
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        # Score 0 means no query term occurs in the chunk at all; don't return it.
        return [(i, float(s)) for i, s in ranked[:k] if s > 0]
