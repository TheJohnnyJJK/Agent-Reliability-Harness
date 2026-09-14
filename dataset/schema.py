"""Data contracts for the fixed support-ops dataset.

Every agent in this benchmark answers questions against the same small,
deterministic set of customers and orders. These models are what
seed_data.py produces and db.py stores - declaring them once here means
the three framework adapters and the tests all agree on the same shape.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

_ID_MAX = 20
_NAME_MAX = 100
_EMAIL_MAX = 200
_ITEM_MAX = 100

AccountStanding = Literal["good", "flagged_fraud", "past_due"]
OrderStatus = Literal["processing", "shipped", "delivered", "cancelled", "refunded"]
RefundReason = Literal[
    "defective", "not_as_described", "changed_mind", "late_delivery", "fraud_suspected"
]


class Customer(BaseModel):
    customer_id: str = Field(min_length=1, max_length=_ID_MAX)
    name: str = Field(min_length=1, max_length=_NAME_MAX)
    email: str = Field(min_length=1, max_length=_EMAIL_MAX)
    account_standing: AccountStanding
    signup_date: str  # ISO date, e.g. "2025-03-14"


class Order(BaseModel):
    order_id: str = Field(min_length=1, max_length=_ID_MAX)
    customer_id: str = Field(min_length=1, max_length=_ID_MAX)
    item: str = Field(min_length=1, max_length=_ITEM_MAX)
    amount_usd: float = Field(ge=0)
    order_date: str  # ISO date
    status: OrderStatus


class RefundPolicy(BaseModel):
    reason: RefundReason
    refund_pct: float = Field(ge=0, le=1)
    max_days_since_order: int = Field(ge=0)
    requires_escalation: bool
