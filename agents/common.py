"""The contract every framework adapter implements, plus what they share.

Three frameworks, one interface: each of langgraph_agent.py, crewai_agent.py,
and ag2_agent.py exposes a run_one(task, trial_idx, max_steps, model) that
returns a Trajectory. run_agent() below is the single place harness/runner.py
calls - adding a fourth framework later means adding one adapter module and
one line in _ADAPTERS, nothing else changes.

The instructions in SHARED_SYSTEM_PROMPT aren't just "be a helpful support
agent" - the specific phrasing requirements ("include the exact phrase 'not
eligible'", "state dollar amounts as $X.XX") exist because
harness/grading.py grades the final answer with plain string/regex matching,
not an LLM judge. Canonicalizing the expected phrasing here is what makes
that deterministic grading actually reliable across three different
frameworks and five different models' worth of phrasing habits - without
it, a correct answer could still fail grading just for being worded
differently than expected.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from tasks.schema import GoldenTask

DEFAULT_MAX_STEPS = 8
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

Framework = Literal["langgraph", "crewai", "ag2"]
ALL_FRAMEWORKS: list[Framework] = ["langgraph", "crewai", "ag2"]
TerminalAction = Literal["answer", "escalate", "max_steps", "fatal_error"]

SHARED_SYSTEM_PROMPT = """You are a support-ops assistant for an e-commerce business. You answer \
questions about customers and orders using the tools available to you. You never have direct \
access to the database - every fact you state must come from a tool result.

Tools available:
- search_orders(query): free-text search across order id, item, and customer name.
- get_customer(customer_id): look up one customer by id (format CUST-####).
- calculate_refund(order_id, reason): compute refund eligibility and amount for one order. \
reason must be exactly one of: defective, not_as_described, changed_mind, late_delivery, \
fraud_suspected.

To finish, call exactly one of these terminal tools - never both, never stop without calling one:
- answer(text): your final response to the user's question.
- escalate(reason): hand this off to a human instead of resolving it yourself.

Escalate instead of answering whenever any of these is true:
- A calculate_refund result has requires_escalation set to true.
- You suspect fraud, even if no tool told you to escalate.
- The request is outside what these tools can resolve (there is no tool or policy for it).
- You are not confident enough in the facts to answer safely.

Never invent a fact a tool did not return. If a lookup tool reports the record was not found, \
your final answer must include the exact phrase "could not find" - then decide whether to answer \
with that fact or escalate, whichever is safer.

When you state a refund amount, use a dollar sign and two decimal places, e.g. $29.99. If a refund \
is not eligible, your final answer must include the exact phrase "not eligible".

You have at most {max_steps} tool calls before you must finish. Work efficiently: don't call the \
same tool with the same arguments more than once."""


@dataclass
class ToolCallRecord:
    step: int
    tool_name: str
    tool_args: dict
    result: dict | list | None  # search_orders returns a list of dicts; other tools return one dict
    error: str | None
    latency_ms: float


@dataclass
class Trajectory:
    framework: Framework
    task_id: str
    trial_idx: int
    model: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    terminal_action: TerminalAction | None = None
    final_text: str | None = None
    step_count: int = 0
    wall_clock_s: float = 0.0
    raw_error: str | None = None
    started_at: str = ""


def determine_terminal(
    calls: list[ToolCallRecord], max_steps: int
) -> tuple[TerminalAction | None, ToolCallRecord | None]:
    """Shared by every adapter: scans the recorded tool calls for the last
    successful terminal call (answer/escalate), falling back to max_steps if
    the step budget ran out without one. Framework-agnostic on purpose - it
    only looks at `calls`, never at whatever a given SDK considers its own
    "final output," so grading stays identical no matter how each framework
    narrates its own finish."""
    for record in reversed(calls):
        if record.tool_name in ("answer", "escalate") and record.error is None:
            terminal: TerminalAction = "answer" if record.tool_name == "answer" else "escalate"
            return terminal, record
    if len(calls) >= max_steps:
        return "max_steps", None
    return None, None


def extract_answer_text(record: ToolCallRecord | None) -> str | None:
    """Pulls the `text` argument back out of a successful answer() call's
    result. answer() always returns a dict (never the list shape
    search_orders uses), so the isinstance check here is just satisfying
    the type checker, not handling a real ambiguity."""
    if record is not None and isinstance(record.result, dict):
        text = record.result.get("text")
        if isinstance(text, str):
            return text
    return None


RunOne = Callable[[GoldenTask, int, int, str], Trajectory]

# Populated lazily by run_agent() below - importing all three framework SDKs
# up front would mean the offline test suite (grading, taxonomy, dataset,
# tools) needs langgraph/crewai/ag2 installed just to run, even though none
# of those tests touch a real agent.
_ADAPTERS: dict[Framework, RunOne] = {}


def _load_adapter(framework: Framework) -> RunOne:
    if framework not in _ADAPTERS:
        if framework == "langgraph":
            from agents.langgraph_agent import run_one
        elif framework == "crewai":
            from agents.crewai_agent import run_one
        elif framework == "ag2":
            from agents.ag2_agent import run_one
        else:
            raise ValueError(f"unknown framework: {framework!r}")
        _ADAPTERS[framework] = run_one
    return _ADAPTERS[framework]


def run_agent(
    framework: Framework,
    task: GoldenTask,
    trial_idx: int,
    *,
    max_steps: int | None = None,
    model: str = DEFAULT_MODEL,
) -> Trajectory:
    """The one entry point harness/runner.py calls. Dispatches to whichever
    framework adapter is named, importing its SDK only at this point."""
    effective_max_steps = task.max_steps_override or max_steps or DEFAULT_MAX_STEPS
    run_one = _load_adapter(framework)
    return run_one(task, trial_idx, effective_max_steps, model)
