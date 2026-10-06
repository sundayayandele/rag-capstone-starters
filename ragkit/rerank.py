"""Second-stage rerankers.

LexicalReranker (default, offline): scores each candidate by question-term coverage of the title and body,
plus a bigram-overlap bonus. A deliberately simple stand-in for a cross-encoder.

CrossEncoderReranker (optional): sentence-transformers CrossEncoder; set RAG_RERANKER=ce, install `.[ml]`.
"""
from __future__ import annotations
import os

from .index import Hit
from .text import coverage, terms


def _bigrams(ts: list[str]) -> set[tuple[str, str]]:
    return set(zip(ts, ts[1:]))


class LexicalReranker:
    name = "lexical"

    def rerank(self, query: str, hits: list[Hit], top: int = 5) -> list[Hit]:
        qt = terms(query)
        qb = _bigrams(qt)
        scored = []
        for h in hits:
            body_cov = coverage(query, h.chunk.text)
            title_cov = coverage(query, h.chunk.title)
            ct = terms(h.chunk.text)
            big = len(qb & _bigrams(ct)) / len(qb) if qb else 0.0
            scored.append(Hit(h.chunk, 0.5 * body_cov + 0.3 * title_cov + 0.2 * big))
        scored.sort(key=lambda x: -x.score)
        return scored[:top]


class CrossEncoderReranker:
    name = "ce"

    def __init__(self, model: str | None = None):
        from sentence_transformers import CrossEncoder  # optional dependency
        self.model = CrossEncoder(model or os.getenv("RAG_CE_MODEL", "BAAI/bge-reranker-base"))

    def rerank(self, query: str, hits: list[Hit], top: int = 5) -> list[Hit]:
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda x: -x[1])[:top]
        return [Hit(h.chunk, float(s)) for h, s in ranked]


def get_reranker():
    return CrossEncoderReranker() if os.getenv("RAG_RERANKER", "lexical").lower() == "ce" else LexicalReranker()
