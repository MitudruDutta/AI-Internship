"""
chroma_store.py — create and manage Chroma vector collections.

A thin wrapper around ChromaDB that handles:
  - a persistent client (data survives between runs)
  - creating / getting a collection with a chosen embedding function
  - ingesting documents (with metadata + ids)
  - retrieving documents by semantic query
  - basic management (count, peek, delete)

Used by the ingestion pipeline (ingest.py) and later by the RAG query layer.
"""

from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION = "pubmed_intermittent_fasting"


class ChromaStore:
    """Manage a single Chroma collection backed by a persistent store."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_function=None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Persistent client: the vector data is written to `persist_dir` on disk.
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Default embeddings = sentence-transformers all-MiniLM-L6-v2 (384-dim),
        # local and free. Pass a custom embedding_function to override.
        self.embedding_function = (
            embedding_function or embedding_functions.DefaultEmbeddingFunction()
        )

        # get_or_create so re-running the pipeline reuses the same collection.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},   # cosine distance for text embeddings
        )

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def add_documents(self, ids, documents, metadatas=None, batch_size: int = 100):
        """
        Add documents to the collection (uses `upsert` so re-ingesting the same
        id updates rather than duplicates). Ingests in batches.

        ids        : list[str]  unique id per document (e.g. the PMID)
        documents  : list[str]  the text to embed
        metadatas  : list[dict] optional metadata per document
        """
        if not ids:
            return 0
        if len(ids) != len(documents):
            raise ValueError("ids and documents must be the same length")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas must match ids length")

        total = 0
        for i in range(0, len(ids), batch_size):
            sl = slice(i, i + batch_size)
            self.collection.upsert(
                ids=ids[sl],
                documents=documents[sl],
                metadatas=metadatas[sl] if metadatas is not None else None,
            )
            total += len(ids[sl])
        return total

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def query(self, query_text: str, n_results: int = 5, where: Optional[dict] = None):
        """
        Semantic search. Returns a list of dicts:
        [{id, document, metadata, distance}, ...] ordered by similarity.
        """
        res = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )
        out = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append({
                "id": ids[i],
                "document": docs[i] if i < len(docs) else None,
                "metadata": metas[i] if i < len(metas) else None,
                "distance": dists[i] if i < len(dists) else None,
            })
        return out

    def get(self, ids):
        """Fetch documents by id (no embedding search)."""
        return self.collection.get(ids=ids)

    # ------------------------------------------------------------------ #
    # Management
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        return self.collection.count()

    def peek(self, n: int = 3):
        return self.collection.peek(limit=n)

    def reset(self):
        """Delete and recreate the collection (drops all stored vectors)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
