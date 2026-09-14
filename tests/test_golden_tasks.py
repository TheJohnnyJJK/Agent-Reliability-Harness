"""Offline, deterministic - validates golden_tasks.json against its own
schema and against the seeded dataset, so this suite never needs an API
key and never makes a network call."""
import re

from dataset.seed_data import build_customers, build_orders
from tasks.schema import load_golden_tasks

_ID_RE = re.compile(r"\b(?:ORD|CUST)-\d{4}\b")


def _known_ids() -> set[str]:
    return {c.customer_id for c in build_customers()} | {o.order_id for o in build_orders()}


def test_loads_and_validates_against_schema():
    tasks = load_golden_tasks()
    assert 20 <= len(tasks) <= 25


def test_task_ids_are_unique():
    tasks = load_golden_tasks()
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))


def test_all_five_categories_are_represented():
    tasks = load_golden_tasks()
    categories = {t.category for t in tasks}
    expected = {"lookup", "refund_calc", "escalation_required", "nonexistent_id", "ambiguous"}
    assert categories == expected


def test_nonexistent_id_tasks_reference_ids_absent_from_the_dataset():
    known = _known_ids()
    for t in load_golden_tasks():
        if t.category != "nonexistent_id":
            continue
        referenced = set(_ID_RE.findall(t.prompt))
        assert referenced, f"{t.task_id} is nonexistent_id but its prompt has no ORD-/CUST- id"
        overlap = referenced & known
        assert not overlap, f"{t.task_id} references an id that actually exists: {overlap}"


def test_non_nonexistent_id_tasks_reference_real_ids():
    known = _known_ids()
    for t in load_golden_tasks():
        if t.category == "nonexistent_id":
            continue
        referenced = set(_ID_RE.findall(t.prompt))
        unknown = referenced - known
        assert not unknown, f"{t.task_id} references an id not in the dataset: {unknown}"


def test_escalation_required_and_ambiguous_expect_escalate():
    for t in load_golden_tasks():
        if t.category in ("escalation_required", "ambiguous"):
            assert t.expected_action == "escalate"
