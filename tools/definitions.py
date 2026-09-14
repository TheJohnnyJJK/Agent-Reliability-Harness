"""The five tools every framework agent shares, as plain Python functions.

Framework-agnostic on purpose - no LangChain/CrewAI/AG2 types anywhere in
these signatures. Each framework adapter in agents/ wraps these in whatever
tool-decorator shape its SDK wants, but the actual logic (and the actual
bugs, if there are any) lives in exactly one place, tested once.

Two of these are "real" tools (search_orders, get_customer, calculate_refund)
and two are terminal actions (answer, escalate) an agent calls to signal
it's done. Modeling "I'm finished" as a tool call, instead of parsing free
text for a final answer, is what lets harness/grading.py treat all three
frameworks identically - LangGraph, CrewAI, and AG2 all surface tool calls
in their own way, but "was answer() or escalate() called, and with what
argument" is the same question no matter which one you're asking.
"""
from __future__ import annotations

import re
from datetime import date

from dataset import db
from dataset.seed_data import REFERENCE_DATE, REFUND_POLICIES

_ORDER_ID_RE = re.compile(r"^ORD-\d{4}$")
_CUSTOMER_ID_RE = re.compile(r"^CUST-\d{4}$")
_KNOWN_REASONS = {p.reason for p in REFUND_POLICIES}


class ToolInputError(ValueError):
    """A tool received a malformed argument - wrong shape or an unknown
    enum value - as opposed to a well-formed argument that legitimately
    resolves to "not found". That distinction is what lets
    harness/taxonomy.py tell a genuine tool-usage mistake (tool_call_error)
    apart from a case an agent is expected to handle gracefully (a real,
    well-formed ID that just isn't in the dataset)."""


def search_orders(query: str) -> list[dict]:
    """Free-text match across order id, item, and customer name. Returns []
    (not an error) when nothing matches, so an agent can branch on "no
    results" without needing to catch an exception for an ordinary case."""
    if not query or not query.strip():
        raise ToolInputError("search_orders: query must be a non-empty string")
    return [o.model_dump() for o in db.search_orders(query.strip())]


def get_customer(customer_id: str) -> dict:
    """Raises ToolInputError if customer_id doesn't look like CUST-####.
    Returns {"found": False, "customer_id": ...} if well-formed but absent
    from the dataset; otherwise the full customer record."""
    if not _CUSTOMER_ID_RE.match(customer_id or ""):
        raise ToolInputError(
            f"get_customer: {customer_id!r} is not a valid customer id (expected CUST-####)"
        )
    customer = db.get_customer(customer_id)
    if customer is None:
        return {"found": False, "customer_id": customer_id}
    return {"found": True, **customer.model_dump()}


def calculate_refund(order_id: str, reason: str) -> dict:
    """Raises ToolInputError if order_id isn't shaped like ORD-#### or reason
    isn't one of the five known policy reasons - reason is a closed enum the
    agent must pick from, so an unrecognized value is a tool-usage mistake,
    not a data problem.

    Returns {"found": False, "order_id": ...} if order_id is well-formed but
    absent. Otherwise returns eligibility + amount + whether this case must
    be escalated instead of auto-resolved. requires_escalation is True
    whenever the policy itself requires it (fraud_suspected) OR the order's
    customer has account_standing == "flagged_fraud", regardless of the
    stated reason - a flagged account overrides whatever reason was given.
    """
    if not _ORDER_ID_RE.match(order_id or ""):
        raise ToolInputError(
            f"calculate_refund: {order_id!r} is not a valid order id (expected ORD-####)"
        )
    if reason not in _KNOWN_REASONS:
        raise ToolInputError(
            f"calculate_refund: {reason!r} is not a known refund reason ({sorted(_KNOWN_REASONS)})"
        )

    order = db.get_order(order_id)
    if order is None:
        return {"found": False, "order_id": order_id}

    customer = db.get_customer(order.customer_id)
    policy = next(p for p in REFUND_POLICIES if p.reason == reason)

    days_since_order = (REFERENCE_DATE - date.fromisoformat(order.order_date)).days
    within_window = days_since_order <= policy.max_days_since_order
    not_cancelled = order.status != "cancelled"
    eligible = within_window and not_cancelled

    flagged_account = customer is not None and customer.account_standing == "flagged_fraud"
    requires_escalation = policy.requires_escalation or flagged_account

    if not eligible:
        note = "order was cancelled" if not not_cancelled else (
            f"{days_since_order} days since order exceeds the "
            f"{policy.max_days_since_order}-day window for '{reason}'"
        )
    elif flagged_account:
        note = "customer account is flagged for fraud review - do not auto-resolve"
    else:
        note = f"eligible for {policy.refund_pct:.0%} refund under the '{reason}' policy"

    return {
        "found": True,
        "order_id": order_id,
        "reason": reason,
        "eligible": eligible,
        "refund_amount_usd": round(order.amount_usd * policy.refund_pct, 2) if eligible else 0.0,
        "requires_escalation": requires_escalation,
        "policy_note": note,
    }


def escalate(reason: str) -> dict:
    """Terminal action: hand this off to a human instead of auto-resolving."""
    if not reason or not reason.strip():
        raise ToolInputError("escalate: reason must be a non-empty string")
    return {"action": "escalate", "reason": reason.strip()}


def answer(text: str) -> dict:
    """Terminal action: this is the agent's final response to the user."""
    if not text or not text.strip():
        raise ToolInputError("answer: text must be a non-empty string")
    return {"action": "answer", "text": text.strip()}
