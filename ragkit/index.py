"""Dense index, BM25 index, reciprocal rank fusion and a hybrid wrapper."""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .chunk import Chunk
from .embed import get_embedder
from .text import terms, with_parts


@dataclass
class Hit:
    chunk: Chunk
    score: float


class DenseIndex:
    def __init__(self, chunks: list[Chunk], embedder=None):
        self.chunks = chunks
        self.embedder = embedder or get_embedder()
        self.matrix = self.embedder.encode([c.text for c in chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(with_parts(terms(c.text))) for c in chunks]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df: dict[str, int] = defaultdict(int)
        for d in self.docs:
            for t in d:
                df[t] += 1
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = with_parts(terms(query))
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            scores.append(s)
        order = np.argsort(-np.asarray(scores))[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def rrf(lists: list[list[Hit]], k: int = 60, top: int = 10) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum 1/(k + rank). Robust to different score scales."""
    agg: dict[str, float] = defaultdict(float)
    keep: dict[str, Chunk] = {}
    for hits in lists:
        for rank, h in enumerate(hits, 1):
            agg[h.chunk.id] += 1.0 / (k + rank)
            keep[h.chunk.id] = h.chunk
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:top]
    return [Hit(keep[cid], s) for cid, s in ranked]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder=None):
        self.dense = DenseIndex(chunks, embedder)
        self.bm25 = BM25Index(chunks)

    def search(self, query: str, k: int = 5, depth: int = 20) -> list[Hit]:
        return rrf([self.bm25.search(query, depth), self.dense.search(query, depth)], top=k)
