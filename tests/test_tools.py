"""Offline, deterministic - exercises the tool functions against a freshly
seeded test database, so this suite never needs an API key and never makes
a network call."""
import os
import tempfile

import pytest

from dataset import db
from tools.definitions import (
    ToolInputError,
    answer,
    calculate_refund,
    escalate,
    get_customer,
    search_orders,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    original_path = db.DB_PATH
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    db.DB_PATH = path
    db.reset_db()
    yield
    if os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)
    db.DB_PATH = original_path


def test_search_orders_matches_item():
    results = search_orders("mouse")
    assert any(r["order_id"] == "ORD-1042" for r in results)


def test_search_orders_empty_query_is_a_tool_error():
    with pytest.raises(ToolInputError):
        search_orders("   ")


def test_search_orders_no_match_returns_empty_list_not_error():
    assert search_orders("nonexistent-widget-zzz") == []


def test_get_customer_found():
    result = get_customer("CUST-1001")
    assert result["found"] is True
    assert result["name"] == "Jane Doe"


def test_get_customer_well_formed_but_absent():
    result = get_customer("CUST-9999")
    assert result == {"found": False, "customer_id": "CUST-9999"}


def test_get_customer_malformed_id_is_a_tool_error():
    with pytest.raises(ToolInputError):
        get_customer("not-an-id")


def test_calculate_refund_eligible_defective():
    result = calculate_refund("ORD-1042", "defective")
    assert result["found"] is True
    assert result["eligible"] is True
    assert result["refund_amount_usd"] == pytest.approx(29.99)
    assert result["requires_escalation"] is False


def test_calculate_refund_flagged_account_forces_escalation():
    # ORD-1077 belongs to CUST-1002, who is flagged_fraud - even a mundane
    # reason must come back requiring escalation.
    result = calculate_refund("ORD-1077", "defective")
    assert result["requires_escalation"] is True


def test_calculate_refund_outside_window_is_not_eligible():
    # ORD-1055 is 60 days old; 'defective' only covers a 30-day window.
    result = calculate_refund("ORD-1055", "defective")
    assert result["eligible"] is False
    assert result["refund_amount_usd"] == 0.0


def test_calculate_refund_cancelled_order_is_not_eligible():
    # ORD-1090 is cancelled - no refund regardless of reason or window.
    result = calculate_refund("ORD-1090", "changed_mind")
    assert result["eligible"] is False


def test_calculate_refund_well_formed_but_absent_order():
    result = calculate_refund("ORD-9999", "defective")
    assert result == {"found": False, "order_id": "ORD-9999"}


def test_calculate_refund_unknown_reason_is_a_tool_error():
    with pytest.raises(ToolInputError):
        calculate_refund("ORD-1042", "buyers_remorse")


def test_calculate_refund_malformed_order_id_is_a_tool_error():
    with pytest.raises(ToolInputError):
        calculate_refund("not-an-order", "defective")


def test_answer_and_escalate_shapes():
    result = answer("the order shipped yesterday")
    assert result == {"action": "answer", "text": "the order shipped yesterday"}
    assert escalate("suspected fraud") == {"action": "escalate", "reason": "suspected fraud"}


def test_answer_empty_text_is_a_tool_error():
    with pytest.raises(ToolInputError):
        answer("")


def test_escalate_empty_reason_is_a_tool_error():
    with pytest.raises(ToolInputError):
        escalate("")
