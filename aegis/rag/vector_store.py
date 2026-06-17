from __future__ import annotations

import re
from collections import Counter
from typing import List

from rag.seed_data import SEED_DOCUMENTS, _chunk_text

# Build corpus once at import time from the hardcoded seed data
_CORPUS: list[dict] = []

for _entry in SEED_DOCUMENTS:
    for _chunk in _chunk_text(_entry["text"]):
        _CORPUS.append({"text": _chunk, "source": _entry["source"]})


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for token in set(query_tokens):
        if token in tf:
            f = tf[token]
            score += (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
    return score


def _rank(query: str, k: int = 4) -> list[tuple[dict, float]]:
    query_tokens = _tokenize(query)
    if not query_tokens or not _CORPUS:
        return []
    doc_tokens = [_tokenize(d["text"]) for d in _CORPUS]
    avgdl = sum(len(t) for t in doc_tokens) / len(doc_tokens)
    scored = [
        (_CORPUS[i], _bm25_score(query_tokens, doc_tokens[i], avgdl))
        for i in range(len(_CORPUS))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(d, s) for d, s in scored[:k] if s > 0]


def retrieve_clinical_context(query: str, k: int = 4) -> List[str]:
    return [d["text"] for d, _ in _rank(query, k)]


def retrieve_with_sources(query: str, k: int = 4) -> List[dict]:
    return [
        {"text": d["text"], "source": d["source"], "score": score}
        for d, score in _rank(query, k)
    ]


def add_documents(texts: List[str], metadatas: List[dict]) -> None:
    for text, meta in zip(texts, metadatas):
        _CORPUS.append({"text": text, "source": meta.get("source_document", "unknown")})


def collection_count() -> int:
    return len(_CORPUS)
