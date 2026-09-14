"""SQLite storage for the fixed customer/order dataset.

Mirrors lead-router's agent/store.py: one connection per call via _conn(),
`?` placeholders everywhere (never string-formatted SQL), AGENT_HARNESS_DB
picks the file. The one real difference - this project's tables are
write-once. seed() runs exactly once to populate them; every tool in
tools/definitions.py only ever SELECTs. No agent under test can write to
its own eval fixture, on purpose (see the Security section in the README).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

from dataset.schema import Customer, Order
from dataset.seed_data import build_customers, build_orders

_DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "harness.db")
DB_PATH = os.environ.get("AGENT_HARNESS_DB", _DEFAULT_DB_PATH)

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    account_standing TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    item TEXT NOT NULL,
    amount_usd REAL NOT NULL,
    order_date TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(SCHEMA)


def seed() -> None:
    """Populates customers/orders from dataset/seed_data.py. Safe to call
    repeatedly - clears both tables first, so re-seeding is idempotent
    rather than accumulating duplicate rows on every test run."""
    init_db()
    with _conn() as conn:
        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM customers")
        conn.executemany(
            "INSERT INTO customers (customer_id, name, email, account_standing, signup_date) "
            "VALUES (?,?,?,?,?)",
            [
                (c.customer_id, c.name, c.email, c.account_standing, c.signup_date)
                for c in build_customers()
            ],
        )
        conn.executemany(
            "INSERT INTO orders (order_id, customer_id, item, amount_usd, order_date, status) "
            "VALUES (?,?,?,?,?,?)",
            [
                (o.order_id, o.customer_id, o.item, o.amount_usd, o.order_date, o.status)
                for o in build_orders()
            ],
        )


def get_customer(customer_id: str) -> Customer | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
    return Customer(**dict(row)) if row else None


def get_order(order_id: str) -> Order | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    return Order(**dict(row)) if row else None


def search_orders(query: str, limit: int = 10) -> list[Order]:
    """Case-insensitive substring match across order_id, item, and the
    owning customer's name - a join, not three separate lookups, since a
    real support search box doesn't know in advance which field the
    customer is going to type into."""
    like = f"%{query.lower()}%"
    with _conn() as conn:
        rows = conn.execute(
            """SELECT orders.* FROM orders
               JOIN customers ON customers.customer_id = orders.customer_id
               WHERE lower(orders.order_id) LIKE ?
                  OR lower(orders.item) LIKE ?
                  OR lower(customers.name) LIKE ?
               ORDER BY orders.order_id LIMIT ?""",
            (like, like, like, limit),
        ).fetchall()
    return [Order(**dict(r)) for r in rows]


def reset_db() -> None:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    seed()
