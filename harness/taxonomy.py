"""Classifies every trial's trajectory into one failure category, or success.

Checks run in this exact order, first match wins:

1. infinite_loop - checked first because a looping run's other symptoms
   (tool errors, unresolved ids) are downstream noise, not the real story.
2. tool_call_error - a tool raised, and the run never recovered to a clean
   terminal action afterward.
3. hallucinated_parameter - an id-shaped tool argument that couldn't have
   been legitimately obtained.
4. otherwise: success if grading.grade() passes, else silent_wrong_answer.

The provenance check in step 3 is what keeps a task like hallucinated-01
(the user supplies a nonexistent order id, and the agent correctly reports
"could not find") from being misclassified as a hallucination - reusing an
id the *prompt itself* supplied is legitimate tool use, not invention.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from agents.common import Trajectory
from dataset.seed_data import build_customers, build_orders
from harness.grading import grade
from tasks.schema import GoldenTask

_REPEAT_THRESHOLD = 3  # same (tool, args) pair appearing >= 3x anywhere in a trajectory = looping

_ID_LIKE_RE = re.compile(r"^(?:ORD|CUST)-\d{4}$")
_ID_IN_TEXT_RE = re.compile(r"\b(?:ORD|CUST)-\d{4}\b")

FailureCategory = Literal[
    "success", "tool_call_error", "infinite_loop", "hallucinated_parameter", "silent_wrong_answer"
]


def _dataset_ids() -> set[str]:
    return {c.customer_id for c in build_customers()} | {o.order_id for o in build_orders()}


def _ids_in_text(text: str) -> set[str]:
    return set(_ID_IN_TEXT_RE.findall(text or ""))


def _ids_in_result(result: object) -> set[str]:
    """Walks a tool result for id-shaped strings. Most tools return a single
    dict, but search_orders returns a list of dicts (each match is its own
    record) - this has to handle both shapes, recursively, since a result
    can nest either inside the other."""
    values: Iterable[object]
    if isinstance(result, dict):
        values = result.values()
    elif isinstance(result, list):
        values = result
    else:
        return set()
    ids: set[str] = set()
    for value in values:
        if isinstance(value, str) and _ID_LIKE_RE.match(value):
            ids.add(value)
        elif isinstance(value, (dict, list)):
            ids |= _ids_in_result(value)
    return ids


def _is_looping(traj: Trajectory) -> bool:
    if traj.terminal_action == "max_steps":
        return True
    signatures = [(c.tool_name, tuple(sorted(c.tool_args.items()))) for c in traj.tool_calls]
    return any(signatures.count(sig) >= _REPEAT_THRESHOLD for sig in set(signatures))


def _has_unrecovered_tool_error(traj: Trajectory) -> bool:
    had_error = any(c.error is not None for c in traj.tool_calls)
    return had_error and traj.terminal_action in (None, "fatal_error")


def _has_hallucinated_parameter(task: GoldenTask, traj: Trajectory) -> bool:
    known_ids = _dataset_ids() | _ids_in_text(task.prompt)
    for call in traj.tool_calls:
        for arg_value in call.tool_args.values():
            if not isinstance(arg_value, str) or not _ID_LIKE_RE.match(arg_value):
                continue
            if arg_value not in known_ids:
                return True
        known_ids |= _ids_in_result(call.result)
    return False


def classify(task: GoldenTask, traj: Trajectory) -> FailureCategory:
    if _is_looping(traj):
        return "infinite_loop"
    if _has_unrecovered_tool_error(traj):
        return "tool_call_error"
    if _has_hallucinated_parameter(task, traj):
        return "hallucinated_parameter"
    if traj.terminal_action in ("answer", "escalate"):
        return "success" if grade(task, traj) else "silent_wrong_answer"
    # Didn't loop, didn't hallucinate, but also never reached a clean
    # terminal action (e.g. traj.raw_error set from a framework-level
    # crash) - still needs a bucket, and "the run failed to complete" is
    # closest in spirit to a tool/execution error.
    return "tool_call_error"
