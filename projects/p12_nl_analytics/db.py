"""Deterministic sample sales database and a read-only, validated SQL executor."""
from __future__ import annotations
import random
import re
import sqlite3

REGIONS = ["Europe", "Africa", "Americas", "Asia"]
SEGMENTS = ["Enterprise", "SMB"]
PRODUCTS = [("Analytics Suite", "Software", 1200.0), ("Data Connector", "Software", 300.0), ("Edge Gateway", "Hardware", 850.0),
            ("Sensor Pack", "Hardware", 240.0), ("Rack Server", "Hardware", 2400.0), ("Onboarding", "Services", 1500.0),
            ("Support Plan", "Services", 600.0), ("Training Day", "Services", 900.0), ("Mobile App", "Software", 150.0),
            ("Backup Appliance", "Hardware", 1800.0)]

SCHEMA_CARDS = [
    ("customers", "Table customers(id, name, region, segment). One row per customer. region in Europe, Africa, Americas, Asia. segment in Enterprise, SMB."),
    ("products", "Table products(id, name, category, unit_price). category in Software, Hardware, Services."),
    ("orders", "Table orders(id, customer_id, product_id, quantity, order_date, status). status in completed, cancelled, pending. order_date is ISO text YYYY-MM-DD. Joins: orders.customer_id = customers.id, orders.product_id = products.id."),
    ("metric:revenue", "Metric revenue = SUM(orders.quantity * products.unit_price) over orders with status = 'completed'. Cancelled and pending orders are excluded."),
]


def build_db(conn: sqlite3.Connection | None = None, seed: int = 7) -> sqlite3.Connection:
    conn = conn or sqlite3.connect(":memory:")
    rnd = random.Random(seed)
    conn.executescript("""
        CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, region TEXT, segment TEXT);
        CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, unit_price REAL);
        CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, quantity INTEGER, order_date TEXT, status TEXT);
    """)
    for i in range(1, 41):
        conn.execute("INSERT INTO customers VALUES (?,?,?,?)", (i, f"Customer {i:02d}", rnd.choice(REGIONS), rnd.choice(SEGMENTS)))
    for i, (n, c, p) in enumerate(PRODUCTS, 1):
        conn.execute("INSERT INTO products VALUES (?,?,?,?)", (i, n, c, p))
    for i in range(1, 401):
        y = rnd.choice([2024, 2025])
        d = f"{y}-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"
        st = rnd.choices(["completed", "cancelled", "pending"], [0.8, 0.12, 0.08])[0]
        conn.execute("INSERT INTO orders VALUES (?,?,?,?,?,?)", (i, rnd.randint(1, 40), rnd.randint(1, len(PRODUCTS)), rnd.randint(1, 6), d, st))
    conn.commit()
    return conn


class SQLRejected(ValueError):
    pass


FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate|load_extension|begin|commit|rollback)\b", re.I)
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def validate_sql(sql: str) -> str:
    """Static checks: one statement, SELECT or WITH only, no write/DDL/pragma keywords. Returns cleaned SQL."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise SQLRejected("empty statement")
    if ";" in s:
        raise SQLRejected("multiple statements are not allowed")
    if "--" in s or "/*" in s:
        raise SQLRejected("comments are not allowed")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise SQLRejected("only SELECT queries are allowed")
    if FORBIDDEN.search(s):
        raise SQLRejected("forbidden keyword")
    return s


def safe_execute(conn: sqlite3.Connection, sql: str, limit: int = 100, max_steps: int = 2_000_000):
    """Validate, then run with an authorizer that only allows reads, a row limit and a step budget."""
    s = validate_sql(sql)

    def authorizer(action, *_):
        return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    steps = {"n": 0}

    def progress():
        steps["n"] += 1000
        return 1 if steps["n"] > max_steps else 0

    conn.set_authorizer(authorizer)
    conn.set_progress_handler(progress, 1000)
    try:
        try:
            cur = conn.execute(f"SELECT * FROM ({s}) LIMIT {int(limit)}")
            return [d[0] for d in cur.description], cur.fetchall()
        except sqlite3.Error as e:
            raise SQLRejected(f"execution error: {e}") from e
    finally:
        conn.set_authorizer(None)
        conn.set_progress_handler(None, 0)
