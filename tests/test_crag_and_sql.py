import sqlite3

import pytest

from projects.p09_compliance_crag.pipeline import CragPipeline, rewrite
from projects.p12_nl_analytics.db import SQLRejected, build_db, safe_execute, validate_sql
from projects.p12_nl_analytics.generator import rule_based
from projects.p12_nl_analytics.pipeline import AnalyticsPipeline, BENIGN_SQL, HOSTILE_SQL


def test_rewrite_expands_acronyms_only():
    assert rewrite("What is the deadline for filing a SAR?") == "What is the deadline for filing a suspicious activity report?"
    assert rewrite("Hello world") == "Hello world"


def test_crag_grades_and_corrects():
    p = CragPipeline()
    a = p.ask("What is the deadline for filing a SAR?")
    assert a.meta["grade"] != "correct" and a.meta.get("corrected") and "5 working days" in a.text
    assert p.ask("How long are backups retained?").meta["grade"] == "correct"


@pytest.mark.parametrize("sql", HOSTILE_SQL)
def test_hostile_sql_is_blocked(sql):
    with pytest.raises(SQLRejected):
        safe_execute(build_db(), sql)


@pytest.mark.parametrize("sql", BENIGN_SQL)
def test_benign_sql_runs(sql):
    cols, rows = safe_execute(build_db(), sql)
    assert rows


def test_row_limit_applies():
    _, rows = safe_execute(build_db(), "SELECT * FROM orders", limit=5)
    assert len(rows) == 5


def test_authorizer_blocks_writes_even_if_validation_is_bypassed():
    conn = build_db()
    conn.set_authorizer(lambda action, *a: sqlite3.SQLITE_OK if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION) else sqlite3.SQLITE_DENY)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM orders")


def test_tables_survive_hostile_questions():
    p = AnalyticsPipeline()
    for q in ["Delete all cancelled orders", "Drop the customers table"]:
        assert p.ask(q)["refused"]
    assert p.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 400


def test_schema_retrieval_finds_revenue_metric():
    ctx = AnalyticsPipeline().schema_context("revenue by region")
    assert any("Metric revenue" in c for c in ctx)


def test_rule_based_refuses_unknown():
    assert rule_based("What is the weather in Lagos?") is None
    assert validate_sql("SELECT 1;") == "SELECT 1"
