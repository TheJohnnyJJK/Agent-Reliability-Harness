"""Builds the fixed customer/order dataset every framework agent is tested against.

Two layers, on purpose. A small CURATED set below has fixed IDs
(CUST-1001, ORD-1042, ...) with properties hand-picked to exercise specific
grading paths - a fraud-flagged account, an order past its refund window, a
cancelled order. golden_tasks.json references these IDs directly, so they
never change shape no matter what. On top of that, a larger FILLER set is
generated from a fixed random seed to give search_orders() something
realistic to search across - bulk data, not test fixtures, so it's fine
for it to be "random" as long as it's the same random every time.

REFERENCE_DATE is deliberately frozen, not datetime.now(). If refund-window
math used the real current date, every "order placed 10 days ago" fixture
would silently age past its window a month from now and golden tasks would
start failing for reasons that have nothing to do with agent quality.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from dataset.schema import Customer, Order, RefundPolicy

SEED = 8402
REFERENCE_DATE = date(2026, 9, 15)

REFUND_POLICIES: list[RefundPolicy] = [
    RefundPolicy(reason="defective", refund_pct=1.0, max_days_since_order=30,
                 requires_escalation=False),
    RefundPolicy(reason="not_as_described", refund_pct=1.0, max_days_since_order=30,
                 requires_escalation=False),
    RefundPolicy(reason="changed_mind", refund_pct=0.5, max_days_since_order=14,
                 requires_escalation=False),
    RefundPolicy(reason="late_delivery", refund_pct=0.75, max_days_since_order=45,
                 requires_escalation=False),
    RefundPolicy(reason="fraud_suspected", refund_pct=0.0, max_days_since_order=0,
                 requires_escalation=True),
]

CURATED_CUSTOMERS: list[Customer] = [
    Customer(customer_id="CUST-1001", name="Jane Doe", email="jane.doe@example.com",
             account_standing="good", signup_date="2024-01-10"),
    Customer(customer_id="CUST-1002", name="John Smith", email="john.smith@example.com",
             account_standing="flagged_fraud", signup_date="2025-06-02"),
    Customer(customer_id="CUST-1003", name="Maria Garcia", email="maria.garcia@example.com",
             account_standing="good", signup_date="2023-11-20"),
    Customer(customer_id="CUST-1004", name="Wei Chen", email="wei.chen@example.com",
             account_standing="past_due", signup_date="2024-08-15"),
]

CURATED_ORDERS: list[Order] = [
    Order(order_id="ORD-1042", customer_id="CUST-1001", item="Wireless Mouse", amount_usd=29.99,
          order_date=str(REFERENCE_DATE - timedelta(days=10)), status="delivered"),
    Order(order_id="ORD-1077", customer_id="CUST-1002", item="4K Monitor", amount_usd=349.99,
          order_date=str(REFERENCE_DATE - timedelta(days=5)), status="delivered"),
    Order(order_id="ORD-1055", customer_id="CUST-1001", item="Mechanical Keyboard",
          amount_usd=89.50, order_date=str(REFERENCE_DATE - timedelta(days=60)),
          status="delivered"),
    Order(order_id="ORD-1063", customer_id="CUST-1003", item="Office Chair", amount_usd=249.00,
          order_date=str(REFERENCE_DATE - timedelta(days=3)), status="shipped"),
    Order(order_id="ORD-1090", customer_id="CUST-1004", item="Standing Desk", amount_usd=459.00,
          order_date=str(REFERENCE_DATE - timedelta(days=20)), status="cancelled"),
]

_FIRST_NAMES = [
    "Alex", "Priya", "Sam", "Yuki", "Omar", "Lena", "Carlos", "Ingrid", "Noah", "Fatima",
]
_LAST_NAMES = [
    "Nguyen", "Patel", "Kowalski", "Silva", "Andersen", "Haddad", "Rossi", "Kim", "Novak", "Diallo",
]
_ITEMS = [
    "USB-C Hub", "Desk Lamp", "Laptop Stand", "Bluetooth Speaker", "Webcam",
    "Noise-Cancelling Headphones", "Ergonomic Mouse Pad", "Portable SSD", "Ring Light",
    "Cable Organizer",
]
_STANDINGS = ["good", "good", "good", "good", "past_due"]  # filler skews toward good standing
_STATUSES = [
    "delivered", "delivered", "delivered", "shipped", "processing", "cancelled", "refunded",
]

_FILLER_CUSTOMER_COUNT = 11  # + 4 curated = 15 total
_FILLER_ORDER_COUNT = 33     # + 5 curated = 38 total


def _filler_customers(rng: random.Random) -> list[Customer]:
    customers = []
    for i in range(_FILLER_CUSTOMER_COUNT):
        cid = f"CUST-{1005 + i}"
        first, last = rng.choice(_FIRST_NAMES), rng.choice(_LAST_NAMES)
        signup_offset = rng.randint(60, 900)
        customers.append(Customer(
            customer_id=cid,
            name=f"{first} {last}",
            email=f"{first.lower()}.{last.lower()}@example.com",
            account_standing=rng.choice(_STANDINGS),
            signup_date=str(REFERENCE_DATE - timedelta(days=signup_offset)),
        ))
    return customers


def _filler_orders(rng: random.Random, customer_ids: list[str]) -> list[Order]:
    orders = []
    for i in range(_FILLER_ORDER_COUNT):
        oid = f"ORD-{1100 + i}"
        order_offset = rng.randint(1, 120)
        orders.append(Order(
            order_id=oid,
            customer_id=rng.choice(customer_ids),
            item=rng.choice(_ITEMS),
            amount_usd=round(rng.uniform(9.99, 499.99), 2),
            order_date=str(REFERENCE_DATE - timedelta(days=order_offset)),
            status=rng.choice(_STATUSES),
        ))
    return orders


def build_customers() -> list[Customer]:
    """Curated rows first, then deterministic filler. Two calls with the same
    SEED always return byte-identical output - pinned by tests/test_dataset.py."""
    rng = random.Random(SEED)
    return [*CURATED_CUSTOMERS, *_filler_customers(rng)]


def build_orders() -> list[Order]:
    # Re-seeding here (rather than sharing one rng across both builders) keeps
    # build_orders() independently deterministic regardless of call order.
    rng = random.Random(SEED + 1)
    all_customer_ids = [c.customer_id for c in build_customers()]
    return [*CURATED_ORDERS, *_filler_orders(rng, all_customer_ids)]
