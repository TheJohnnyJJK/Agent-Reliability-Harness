"""Offline, deterministic - exercises dataset generation and SQLite storage
only, so this suite never needs an API key and never makes a network call."""
import os
import tempfile

from dataset import db
from dataset.seed_data import CURATED_CUSTOMERS, CURATED_ORDERS, build_customers, build_orders


def test_build_customers_is_deterministic():
    first = [c.model_dump() for c in build_customers()]
    second = [c.model_dump() for c in build_customers()]
    assert first == second


def test_build_orders_is_deterministic():
    assert [o.model_dump() for o in build_orders()] == [o.model_dump() for o in build_orders()]


def test_expected_row_counts():
    assert len(build_customers()) == 15
    assert len(build_orders()) == 38


def test_curated_ids_are_present():
    customer_ids = {c.customer_id for c in build_customers()}
    order_ids = {o.order_id for o in build_orders()}
    for c in CURATED_CUSTOMERS:
        assert c.customer_id in customer_ids
    for o in CURATED_ORDERS:
        assert o.order_id in order_ids


def test_no_duplicate_ids():
    customer_ids = [c.customer_id for c in build_customers()]
    order_ids = [o.order_id for o in build_orders()]
    assert len(customer_ids) == len(set(customer_ids))
    assert len(order_ids) == len(set(order_ids))


def test_every_order_customer_id_exists():
    customer_ids = {c.customer_id for c in build_customers()}
    for o in build_orders():
        assert o.customer_id in customer_ids


def test_ord_9999_does_not_exist():
    # golden_tasks.json relies on this order ID being absent, to test that
    # agents report "not found" instead of inventing details.
    order_ids = {o.order_id for o in build_orders()}
    assert "ORD-9999" not in order_ids


def _fresh_db_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


def test_seed_and_query_round_trip():
    original_path = db.DB_PATH
    db.DB_PATH = _fresh_db_path()
    try:
        db.reset_db()
        jane = db.get_customer("CUST-1001")
        assert jane is not None
        assert jane.name == "Jane Doe"
        assert db.get_customer("CUST-9999") is None

        mouse_order = db.get_order("ORD-1042")
        assert mouse_order is not None
        assert mouse_order.customer_id == "CUST-1001"
        assert db.get_order("ORD-9999") is None

        results = db.search_orders("mouse")
        assert any(o.order_id == "ORD-1042" for o in results)
    finally:
        if os.path.exists(db.DB_PATH):
            os.remove(db.DB_PATH)
        db.DB_PATH = original_path


def test_reseeding_is_idempotent():
    original_path = db.DB_PATH
    db.DB_PATH = _fresh_db_path()
    try:
        db.reset_db()
        db.seed()
        db.seed()
        assert db.get_customer("CUST-1001") is not None
    finally:
        if os.path.exists(db.DB_PATH):
            os.remove(db.DB_PATH)
        db.DB_PATH = original_path
