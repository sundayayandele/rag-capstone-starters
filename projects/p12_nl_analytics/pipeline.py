"""12.1 Natural-language analytics (Structured RAG / Text-to-SQL): schema retrieval -> SQL -> validate -> read-only execute."""
from __future__ import annotations
import json
import time
from pathlib import Path

from ragkit.chunk import Chunk, load_jsonl
from ragkit.evaluate import mean, p95
from ragkit.index import HybridIndex
from ragkit.llm import get_llm

from .db import SCHEMA_CARDS, SQLRejected, build_db, safe_execute
from .generator import generate_sql

HERE = Path(__file__).parent
PROJECT = "p12_nl_analytics"
TITLE = "12.1 Natural-language analytics (Structured RAG / Text-to-SQL)"
THRESHOLDS = {"execution_accuracy": 0.90, "refusal_accuracy": 0.90, "safety_block_rate": 1.0, "false_block_rate": ("max", 0.0), "p95_latency_s": ("max", 6.0)}

HOSTILE_SQL = [
    "DROP TABLE orders", "DELETE FROM customers", "UPDATE products SET unit_price = 0", "SELECT 1; DROP TABLE orders",
    "INSERT INTO customers VALUES (999,'x','Europe','SMB')", "ATTACH DATABASE 'x.db' AS x", "PRAGMA table_info(orders)",
    "SELECT load_extension('evil')", "SELECT * FROM orders -- ; DROP TABLE orders", "WITH x AS (SELECT 1) DELETE FROM orders",
]
BENIGN_SQL = ["SELECT COUNT(*) FROM orders", "SELECT region, COUNT(*) FROM customers GROUP BY region",
              "WITH t AS (SELECT * FROM products) SELECT name FROM t WHERE category = 'Software'"]


class AnalyticsPipeline:
    def __init__(self, llm=None):
        self.conn = build_db()
        self.llm = llm
        self.cards = [Chunk(id=n, doc=n, text=t, title=n) for n, t in SCHEMA_CARDS]
        self.schema_index = HybridIndex(self.cards)

    def schema_context(self, question: str, k: int = 3) -> list[str]:
        return [h.chunk.text for h in self.schema_index.search(question, k)]

    def ask(self, question: str) -> dict:
        sql = generate_sql(question, self.schema_context(question), self.llm)
        if sql is None:
            return {"refused": True, "sql": None, "rows": None, "error": None}
        try:
            cols, rows = safe_execute(self.conn, sql)
            return {"refused": False, "sql": sql, "columns": cols, "rows": rows, "error": None}
        except SQLRejected as e:
            return {"refused": True, "sql": sql, "rows": None, "error": str(e)}


def _norm(rows):
    return sorted(tuple(round(v, 2) if isinstance(v, float) else v for v in r) for r in rows)


def golden():
    return load_jsonl(HERE / "data" / "golden.jsonl")


def run() -> dict:
    pipe = AnalyticsPipeline(get_llm())
    ok, ref, lat, fails = [], [], [], []
    for g in golden():
        t0 = time.perf_counter()
        out = pipe.ask(g["question"])
        lat.append(time.perf_counter() - t0)
        if g.get("gold_sql"):
            _, gold_rows = safe_execute(pipe.conn, g["gold_sql"], limit=1000)
            good = (not out["refused"]) and _norm(out["rows"]) == _norm(gold_rows)
            ok.append(1.0 if good else 0.0)
            ref.append(0.0 if out["refused"] else 1.0)
            if not good:
                fails.append({"q": g["question"], "gold": [g["gold_sql"]], "got": [out["sql"]], "refused": out["refused"]})
        else:
            ref.append(1.0 if out["refused"] else 0.0)
            if not out["refused"]:
                fails.append({"q": g["question"], "gold": [], "got": [out["sql"]], "refused": False})
    blocked = 0
    for s in HOSTILE_SQL:
        try:
            safe_execute(pipe.conn, s)
        except SQLRejected:
            blocked += 1
    false_blocks = 0
    for s in BENIGN_SQL:
        try:
            safe_execute(pipe.conn, s)
        except SQLRejected:
            false_blocks += 1
    m = {"n_questions": len(ref), "execution_accuracy": mean(ok), "refusal_accuracy": mean(ref),
         "safety_block_rate": round(blocked / len(HOSTILE_SQL), 3), "false_block_rate": round(false_blocks / len(BENIGN_SQL), 3),
         "p95_latency_s": round(p95(lat), 4), "failures": fails[:8]}
    failed = [f"{k} = {m[k]}" for k, t in THRESHOLDS.items() if (m[k] > t[1] if isinstance(t, tuple) else m[k] < t)]
    cols = [("execution_accuracy", "Exec. acc."), ("refusal_accuracy", "Refusal"), ("safety_block_rate", "Hostile SQL blocked"),
            ("false_block_rate", "False blocks"), ("p95_latency_s", "p95 s")]
    return {"project": PROJECT, "title": TITLE, "main": "text-to-SQL", "thresholds": THRESHOLDS, "configs": {"text-to-SQL": m},
            "columns": cols, "failed": failed, "passed": not failed,
            "notes": "Offline mode uses a rule-based baseline that supports a handful of question shapes; set RAG_LLM to evaluate an LLM generator. Execution accuracy compares result sets with gold SQL."}


def make(**kw):
    return AnalyticsPipeline(get_llm(), **kw)
