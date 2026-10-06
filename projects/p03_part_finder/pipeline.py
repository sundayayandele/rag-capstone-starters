"""03.1 Part-Code Finder (Hybrid RAG): BM25 + dense fused with Reciprocal Rank Fusion."""
from __future__ import annotations
import json
from pathlib import Path

from ragkit.answer import answer_from_hits
from ragkit.chunk import Chunk, load_jsonl
from ragkit.evaluate import evaluate_configs
from ragkit.index import BM25Index, DenseIndex, HybridIndex
from ragkit.llm import get_llm

HERE = Path(__file__).parent
PROJECT = "p03_part_finder"
TITLE = "03.1 Part-code finder (Hybrid RAG)"
THRESHOLDS = {"recall@5": 0.90, "precision@1": 0.80, "answer_correctness": 0.80, "citation_accuracy": 0.90, "refusal_accuracy": 0.90,
              "p95_latency_s": ("max", 6.0)}


def load_parts() -> list[Chunk]:
    chunks = []
    for r in load_jsonl(HERE / "data" / "parts.jsonl"):
        body = f"{r['description']}"
        chunks.append(Chunk(id=r["code"], doc=r["code"], text=f"{r['code']} {r['name']}. {body}", title=f"{r['code']} {r['name']}",
                            meta={"heading": r["name"], "body": f"{r['code']} {r['name']}. {body}"}))
    return chunks


class HybridPipeline:
    def __init__(self, mode: str = "hybrid", k: int = 5, rrf_depth: int = 20, llm=None):
        self.chunks = load_parts()
        self.mode = mode
        self.index = {"dense": DenseIndex, "bm25": BM25Index, "hybrid": HybridIndex}[mode](self.chunks)
        self.k, self.depth, self.llm = k, rrf_depth, llm

    def retrieve(self, q: str):
        if self.mode == "hybrid":
            return self.index.search(q, self.k, self.depth)
        return self.index.search(q, self.k)

    def ask(self, question: str):
        return answer_from_hits(question, self.retrieve(question), self.llm, min_coverage=0.6, max_sentences=4)


def golden():
    return load_jsonl(HERE / "data" / "golden.jsonl")


def run() -> dict:
    llm = get_llm()
    pipes = {m: HybridPipeline(m, llm=llm) for m in ("bm25", "dense", "hybrid")}
    return evaluate_configs(PROJECT, TITLE, pipes, golden(), "hybrid", THRESHOLDS,
                            notes="Exact part codes favour keyword search; descriptive queries favour dense search; hybrid fuses both with RRF.")


def make(**kw):
    return HybridPipeline("hybrid", llm=get_llm(), **kw)
