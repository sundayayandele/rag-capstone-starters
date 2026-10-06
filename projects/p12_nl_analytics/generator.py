"""NL to SQL. Offline baseline: a small rule-based generator for common analytics shapes.
With RAG_LLM set, an LLM writes the SQL from retrieved schema and metric cards. Unsupported questions return None (refuse)."""
from __future__ import annotations
import re

from ragkit.llm import get_llm

REV = "SUM(o.quantity * p.unit_price)"
REV_FROM = "FROM orders o JOIN products p ON p.id = o.product_id JOIN customers c ON c.id = o.customer_id"
DESTRUCTIVE = re.compile(r"\b(delete|drop|update|insert|truncate|alter|remove all|wipe)\b", re.I)


def _year(q: str):
    m = re.search(r"\b(2024|2025)\b", q)
    return m.group(1) if m else None


def rule_based(question: str) -> str | None:
    q = question.lower().strip().rstrip("?")
    if DESTRUCTIVE.search(q):
        return None
    y = _year(q)
    yf = f" AND substr(o.order_date,1,4) = '{y}'" if y else ""
    m = re.search(r"top (\d+) (customers|products) by revenue", q)
    if m:
        n, what = int(m.group(1)), m.group(2)
        col, grp = ("c.name", "c.name") if what == "customers" else ("p.name", "p.name")
        return f"SELECT {col}, {REV} AS revenue {REV_FROM} WHERE o.status = 'completed'{yf} GROUP BY {grp} ORDER BY revenue DESC LIMIT {n}"
    if "revenue" in q:
        dim = None
        if "by region" in q or "per region" in q:
            dim = "c.region"
        elif "by category" in q or "per category" in q or "by product category" in q:
            dim = "p.category"
        elif "by segment" in q:
            dim = "c.segment"
        if "category" in q and ("most" in q or "highest" in q) and dim is None:
            return f"SELECT p.category, {REV} AS revenue {REV_FROM} WHERE o.status = 'completed'{yf} GROUP BY p.category ORDER BY revenue DESC LIMIT 1"
        if dim:
            return f"SELECT {dim}, {REV} AS revenue {REV_FROM} WHERE o.status = 'completed'{yf} GROUP BY {dim}"
        return f"SELECT {REV} AS revenue {REV_FROM} WHERE o.status = 'completed'{yf}"
    if re.search(r"how many customers", q):
        return "SELECT COUNT(*) FROM customers"
    if re.search(r"how many .*orders", q) or "number of orders" in q:
        st = next((s for s in ("completed", "cancelled", "pending") if s in q), None)
        where = []
        if st:
            where.append(f"o.status = '{st}'")
        if y:
            where.append(f"substr(o.order_date,1,4) = '{y}'")
        w = (" WHERE " + " AND ".join(where)) if where else ""
        if "per region" in q or "by region" in q:
            return f"SELECT c.region, COUNT(*) FROM orders o JOIN customers c ON c.id = o.customer_id{w} GROUP BY c.region"
        return f"SELECT COUNT(*) FROM orders o{w}"
    if "average order quantity" in q or "average quantity" in q:
        return "SELECT AVG(quantity) FROM orders"
    m = re.search(r"products in the (software|hardware|services) category", q)
    if m:
        return f"SELECT name FROM products WHERE category = '{m.group(1).capitalize()}'"
    return None


LLM_SYSTEM = ("You translate questions into a single read-only SQLite SELECT statement using ONLY the schema and metric definitions provided. "
              "Return only SQL, no explanation. If the question cannot be answered from the schema, or asks to change data, return exactly: REFUSE")


def llm_sql(question: str, cards: list[str], llm) -> str | None:
    out = llm.complete(LLM_SYSTEM, "Schema and metrics:\n" + "\n".join(cards) + f"\n\nQuestion: {question}\nSQL:")
    out = re.sub(r"^```(?:sql)?|```$", "", out.strip(), flags=re.M).strip()
    return None if out.upper().startswith("REFUSE") else out


def generate_sql(question: str, cards: list[str] | None = None, llm=None) -> str | None:
    llm = llm if llm is not None else get_llm()
    if llm is None:
        return rule_based(question)
    if DESTRUCTIVE.search(question):
        return None
    return llm_sql(question, cards or [], llm)
