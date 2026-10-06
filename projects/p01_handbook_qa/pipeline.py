"""01.1 Employee Handbook Q&A Bot (Naive RAG): chunk -> embed -> top-k dense -> grounded answer."""
from __future__ import annotations
from pathlib import Path

from ragkit.answer import answer_from_hits, dedup_docs
from ragkit.chunk import load_jsonl, load_markdown_dir
from ragkit.evaluate import evaluate_configs
from ragkit.index import DenseIndex
from ragkit.llm import get_llm

HERE = Path(__file__).parent
PROJECT = "p01_handbook_qa"
TITLE = "01.1 Handbook Q&A bot (Naive RAG)"
THRESHOLDS = {"recall@5": 0.70, "answer_correctness": 0.70, "citation_accuracy": 0.80, "refusal_accuracy": 0.80, "p95_latency_s": ("max", 6.0)}


class NaivePipeline:
    def __init__(self, k: int = 5, min_coverage: float = 0.6, llm=None):
        self.chunks = load_markdown_dir(HERE / "data" / "corpus")
        self.index = DenseIndex(self.chunks)
        self.k, self.min_coverage, self.llm = k, min_coverage, llm

    def ask(self, question: str):
        hits = self.index.search(question, self.k)
        return answer_from_hits(question, hits, self.llm, self.min_coverage)


def golden():
    return load_jsonl(HERE / "data" / "golden.jsonl")


def run() -> dict:
    llm = get_llm()
    pipes = {"naive k=3": NaivePipeline(k=3, llm=llm), "naive k=5": NaivePipeline(k=5, llm=llm)}
    return evaluate_configs(PROJECT, TITLE, pipes, golden(), "naive k=5", THRESHOLDS,
                            notes="Baseline for the capstone ladder: one dense retrieval, one grounded answer, refusal when support is weak.")


def make(**kw):
    return NaivePipeline(llm=get_llm(), **kw)
