"""Build the persistent ChromaDB index from AST chunks.

Run:  uv run python -m ingest.index            (rebuilds from scratch)

ChromaDB is the single source of truth for the corpus: the dense retriever
queries it directly, and the BM25 index is rebuilt in memory from the same
stored documents at startup (see retrieval/hybrid.py). That way the two
retrievers can never disagree about which chunks exist.
"""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from ingest.chunker import chunk_repo

ROOT = Path(__file__).resolve().parent.parent
REPO_DIR = ROOT / "data" / "requests"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "requests_code"
EMBED_MODEL = "nomic-embed-text"


class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text was trained with task prefixes; without them, query and
    document vectors land in slightly different regions and retrieval suffers.
    OllamaEmbeddings doesn't add them, so we do."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")


def get_embeddings() -> NomicEmbeddings:
    return NomicEmbeddings(model=EMBED_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        # Cosine distance instead of Chroma's default L2: embeddings are
        # compared by direction, not magnitude.
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(batch_size: int = 32) -> int:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    chunks = chunk_repo(REPO_DIR)
    # page_content = header + code (what gets embedded); the raw code lives in
    # metadata so the generator can show/cite it without the synthetic header.
    docs = [
        Document(page_content=c.text, metadata={**c.to_metadata(), "code": c.code})
        for c in chunks
    ]
    store = get_vectorstore()
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        store.add_documents(batch, ids=[d.metadata["id"] for d in batch])
        print(f"  embedded {min(i + batch_size, len(docs))}/{len(docs)}")
    return len(docs)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into {CHROMA_DIR} (collection '{COLLECTION}')")
