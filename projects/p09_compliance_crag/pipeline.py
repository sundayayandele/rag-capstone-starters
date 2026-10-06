"""09.1 Compliance Q&A with Corrective RAG (CRAG): retrieve, grade, rewrite, fall back to a trusted source."""
from __future__ import annotations
import json
import re
from pathlib import Path

from ragkit.answer import REFUSAL, Answer, answer_from_hits, dedup_docs
from ragkit.chunk import load_jsonl, load_markdown_dir
from ragkit.evaluate import COLS, evaluate_configs
from ragkit.index import HybridIndex, Hit
from ragkit.llm import get_llm
from ragkit.text import coverage

HERE = Path(__file__).parent
PROJECT = "p09_compliance_crag"
TITLE = "09.1 Compliance Q&A with Corrective RAG"
THRESHOLDS = {"recall@5": 0.80, "answer_correctness": 0.80, "citation_accuracy": 0.85, "refusal_accuracy": 0.85,
              "p95_latency_s": ("max", 8.0)}
GLOSSARY = json.loads((HERE / "data" / "glossary.json").read_text())

CORRECT, AMBIGUOUS, INCORRECT = "correct", "ambiguous", "incorrect"


def rewrite(question: str) -> str:
    """Offline rewrite: expand acronyms with the domain glossary. With an LLM, replace by a rewrite prompt."""
    def sub(m):
        return GLOSSARY.get(m.group(0).lower(), m.group(0))
    return re.sub(r"\b[A-Za-z]{2,5}\b", lambda m: sub(m) if m.group(0).lower() in GLOSSARY else m.group(0), question)


class CragPipeline:
    def __init__(self, correct: bool = True, use_fallback: bool = True, llm=None, hi: float = 0.6, lo: float = 0.3):
        self.primary = HybridIndex(load_markdown_dir(HERE / "data" / "corpus"))
        self.fallback = HybridIndex(load_markdown_dir(HERE / "data" / "fallback", prefix=""))
        self.correct, self.use_fallback, self.llm, self.hi, self.lo = correct, use_fallback, llm, hi, lo

    def grade(self, question: str, hits: list[Hit]) -> str:
        best = max((coverage(question, h.chunk.text) for h in hits[:3]), default=0.0)
        return CORRECT if best >= self.hi else AMBIGUOUS if best >= self.lo else INCORRECT

    def ask(self, question: str) -> Answer:
        hits = self.primary.search(question, 5)
        if not self.correct:
            return answer_from_hits(question, hits, self.llm, self.hi, meta={"grade": "n/a"})
        grade = self.grade(question, hits)
        if grade == CORRECT:
            return answer_from_hits(question, hits, self.llm, self.hi, meta={"grade": grade})
        q2 = rewrite(question)
        cand = self.primary.search(q2, 5)
        if self.use_fallback:
            cand += self.fallback.search(q2, 5)
        cand.sort(key=lambda h: -coverage(q2, h.chunk.text))
        ans = answer_from_hits(q2, cand, self.llm, self.hi, meta={"grade": grade, "rewritten": q2, "corrected": True})
        ans.retrieved = dedup_docs(cand)
        return ans


def golden():
    return load_jsonl(HERE / "data" / "golden.jsonl")


def run() -> dict:
    llm = get_llm()
    pipes = {
        "naive (no correction)": CragPipeline(correct=False, llm=llm),
        "crag, primary only": CragPipeline(True, use_fallback=False, llm=llm),
        "crag + trusted fallback": CragPipeline(True, True, llm=llm),
    }
    cols = COLS[:2] + COLS[3:7]
    r = evaluate_configs(PROJECT, TITLE, pipes, golden(), "crag + trusted fallback", THRESHOLDS,
                         notes="Questions with acronyms or topics outside the internal corpus fail a single retrieval; the grader triggers a glossary rewrite and a trusted fallback source.")
    return r


def make(**kw):
    return CragPipeline(True, True, llm=get_llm(), **kw)
