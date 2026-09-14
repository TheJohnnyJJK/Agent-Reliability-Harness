"""Deterministic pass/fail grading - no LLM judge, ever.

Every task category reduces to one question: does the trajectory's
terminal_action match what the task expects, and (only when the expected
action is "answer") does the final text satisfy the task's match_type.
Escalation-expected tasks (escalation_required, ambiguous) are graded purely
on terminal_action equality - *why* the agent escalated is documentation
(task.notes), not a grading input. That keeps grading mechanical, matching
lead-router's "grade with code, not vibes" eval philosophy.
"""
from __future__ import annotations

import re

from agents.common import Trajectory
from tasks.schema import GoldenTask

# Prefer a dollar-prefixed number if the answer has one (agents are
# instructed to format refund amounts as $X.XX) - falling straight to "the
# first number anywhere in the text" would grab something like the "10" in
# "ordered 10 days ago" before ever reaching the actual refund figure.
_DOLLAR_NUMBER_RE = re.compile(r"\$\s*(-?\d[\d,]*\.?\d*)")
_BARE_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_number(text: str) -> float | None:
    m = _DOLLAR_NUMBER_RE.search(text) or _BARE_NUMBER_RE.search(text)
    if not m:
        return None
    digits = m.group(1) if m.re is _DOLLAR_NUMBER_RE else m.group(0)
    try:
        return float(digits.replace(",", ""))
    except ValueError:
        return None


def _match_answer(task: GoldenTask, final_text: str | None) -> bool:
    if final_text is None:
        return False
    if task.match_type == "exact":
        return _normalize(final_text) == _normalize(task.expected_answer or "")
    if task.match_type == "numeric_tolerance":
        value = _extract_number(final_text)
        if value is None or task.expected_answer is None:
            return False
        return abs(value - float(task.expected_answer)) <= (task.tolerance or 0)
    if task.match_type == "contains_all":
        norm = _normalize(final_text)
        return all(_normalize(s) in norm for s in (task.required_substrings or []))
    raise ValueError(f"unknown match_type: {task.match_type!r}")


def grade(task: GoldenTask, traj: Trajectory) -> bool:
    if traj.terminal_action != task.expected_action:
        return False
    if task.expected_action == "escalate":
        return True
    return _match_answer(task, traj.final_text)
