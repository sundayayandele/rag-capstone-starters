"""Embedders.

hash (default): deterministic hashing embedder over words and character trigrams with a small synonym
lexicon. No downloads, works offline in CI and on a phone. Good enough to exercise the pipeline and
the evaluation harness; it is NOT a substitute for a neural model.

st: sentence-transformers model (for example BAAI/bge-small-en-v1.5). Install with `pip install -e .[ml]`
and set RAG_EMBEDDER=st (optionally RAG_ST_MODEL=<model name>).
"""
from __future__ import annotations
import math
import os
import re
import zlib
from collections import Counter

import numpy as np

from .text import PART_SPLIT, terms


class HashingEmbedder:
    name = "hash"

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _features(self, text: str) -> Counter:
        feats: Counter = Counter()
        for t in terms(text):
            pieces = [t] + ([p for p in PART_SPLIT.split(t) if p] if PART_SPLIT.search(t) else [])
            for p in pieces:
                feats["w:" + p] += 1.0
                padded = f"#{p}#"
                for i in range(len(padded) - 2):
                    feats["c:" + padded[i:i + 3]] += 0.25
        return feats

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            for f, v in self._features(text).items():
                h = zlib.crc32(f.encode())
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[r, h % self.dim] += sign * (1.0 + math.log(v) if v >= 1 else v)
            n = np.linalg.norm(out[r])
            if n > 0:
                out[r] /= n
        return out


class STEmbedder:
    name = "st"

    def __init__(self, model: str | None = None):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(model or os.getenv("RAG_ST_MODEL", "BAAI/bge-small-en-v1.5"))

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def get_embedder(name: str | None = None):
    name = (name or os.getenv("RAG_EMBEDDER", "hash")).lower()
    if name == "st":
        return STEmbedder()
    return HashingEmbedder()
