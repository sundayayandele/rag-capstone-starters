"""02.1 IT Ticket Search with Reranking: wide first-stage retrieval, then a second-stage reranker."""
from __future__ import annotations
from pathlib import Path

from ragkit.answer import answer_from_hits
from ragkit.chunk import load_jsonl, load_markdown_dir
from ragkit.evaluate import evaluate_configs
from ragkit.index import BM25Index, DenseIndex, HybridIndex
from ragkit.llm import get_llm
from ragkit.rerank import get_reranker

HERE = Path(__file__).parent
PROJECT = "p02_ticket_rerank"
TITLE = "02.1 IT ticket search with reranking (Retrieve-and-Rerank)"
THRESHOLDS = {"recall@5": 0.80, "precision@1": 0.60, "answer_correctness": 0.70, "citation_accuracy": 0.80, "refusal_accuracy": 0.80,
              "p95_latency_s": ("max", 6.0)}


class RerankPipeline:
    """first stage: 'dense' | 'bm25' | 'hybrid'; optional second-stage reranker over a wide candidate list."""

    def __init__(self, first: str = "dense", rerank: bool = False, depth: int = 20, k: int = 5, llm=None):
        self.chunks = load_markdown_dir(HERE / "data" / "corpus")
        self.first = {"dense": DenseIndex, "bm25": BM25Index, "hybrid": HybridIndex}[first](self.chunks)
        self.reranker = get_reranker() if rerank else None
        self.depth, self.k, self.llm = depth, k, llm

    def ask(self, question: str):
        hits = self.first.search(question, self.depth)
        if self.reranker:
            hits = self.reranker.rerank(question, hits, top=self.k)
        else:
            hits = hits[: self.k]
        return answer_from_hits(question, hits, self.llm, min_coverage=0.6)


def golden():
    return load_jsonl(HERE / "data" / "golden.jsonl")


def run() -> dict:
    llm = get_llm()
    pipes = {
        "dense only": RerankPipeline("dense", False, llm=llm),
        "bm25 only": RerankPipeline("bm25", False, llm=llm),
        "dense + rerank": RerankPipeline("dense", True, llm=llm),
        "hybrid + rerank": RerankPipeline("hybrid", True, llm=llm),
    }
    return evaluate_configs(PROJECT, TITLE, pipes, golden(), "hybrid + rerank", THRESHOLDS,
                            notes="Compare first-stage only against first-stage plus reranker. Look at P@1 and MRR: reranking changes order, not the candidate pool.")


def make(**kw):
    return RerankPipeline("hybrid", True, llm=get_llm(), **kw)
