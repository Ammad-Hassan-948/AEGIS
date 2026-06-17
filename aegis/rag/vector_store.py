from __future__ import annotations

import os
import threading
import uuid
from typing import List

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Persistent directory for ChromaDB — sits alongside this file
_CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
_COLLECTION_NAME = "aegis_clinical"

# Module-level singletons — initialised lazily
_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None
_ef = DefaultEmbeddingFunction()  # all-MiniLM-L6-v2 via ONNX, no API key needed

# ONNX runtime is NOT thread-safe for concurrent inference — serialize all queries
_query_lock = threading.Lock()


def get_collection() -> chromadb.Collection:
    """Return (and lazily initialise) the singleton ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        with _query_lock:
            if _collection is None:
                _client = chromadb.PersistentClient(path=_CHROMA_PERSIST_DIR)
                _collection = _client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    embedding_function=_ef,
                    metadata={"hnsw:space": "cosine"},
                )
    return _collection


def retrieve_clinical_context(query: str, k: int = 4) -> List[str]:
    """Return the top-k relevant chunks as plain strings.

    Used by agents that only need raw text for grounding their LLM call.
    """
    col = get_collection()
    with _query_lock:
        results = col.query(query_texts=[query], n_results=min(k, col.count()))
    return results["documents"][0] if results["documents"] else []


def retrieve_with_sources(query: str, k: int = 4) -> List[dict]:
    """Return the top-k relevant chunks with their source document names.

    Each entry:
        {
            "text":   str,   # chunk content
            "source": str,   # e.g. "BNF_81_Appendix_1_Interactions"
            "score":  float  # cosine distance (lower = more similar)
        }

    Agents must copy verbatim substrings from 'text' into their rag_evidence
    fields — the Safety Gate verifies these as substrings of the retrieved text.
    """
    col = get_collection()
    with _query_lock:
        results = col.query(
            query_texts=[query],
            n_results=min(k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    return [
        {
            "text": doc,
            "source": meta.get("source_document", "unknown"),
            "score": float(dist),
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def add_documents(texts: List[str], metadatas: List[dict]) -> None:
    """Add pre-chunked documents to the store (used by seed_vector_store)."""
    col = get_collection()
    ids = [str(uuid.uuid4()) for _ in texts]
    col.add(documents=texts, metadatas=metadatas, ids=ids)


def collection_count() -> int:
    """Return the number of chunks currently stored."""
    return get_collection().count()
