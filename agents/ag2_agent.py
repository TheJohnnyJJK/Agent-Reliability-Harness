"""AG2 adapter: a single Agent with tools, driven through its async ask()
API - AG2 1.0's redesigned core (ag2.Agent / ag2.tool), a clean break from
the old ConversableAgent-based 0.x line this project was originally
planned against. (Worth flagging: AG2 1.0 also dropped the legacy `import
autogen` alias entirely - this version imports as `ag2`, full stop.)
run_one() itself stays synchronous, matching the other two adapters'
interface and harness/runner.py's plain synchronous loop, so it drives the
coroutine with asyncio.run() rather than exposing async up the call stack.

Two things this adapter has to do that the other two don't, because of
gaps in AG2's current single-agent Agent.ask() API at this version:
- There's no built-in step/iteration cap on Agent.ask() the way LangGraph
  has tools_condition + a graph step count, or CrewAI has max_iter. The
  tool wrappers enforce max_steps themselves: once the budget's spent, any
  non-terminal tool call is refused with an error telling the agent to
  finish now, instead of doing the real lookup.
- asyncio.wait_for() wraps the whole call as a hard wall-clock backstop, in
  case a model ignores that refusal and the run genuinely can't terminate.
  A timeout is treated the same as a spent step budget - a run that didn't
  finish, not a framework crash.
"""
from __future__ import annotations

import asyncio
import time

import ag2
from ag2.config import AnthropicConfig

from agents.common import (
    SHARED_SYSTEM_PROMPT,
    ToolCallRecord,
    Trajectory,
    determine_terminal,
    extract_answer_text,
)
from tasks.schema import GoldenTask
from tools import definitions as tool_impl

_WALL_CLOCK_TIMEOUT_S = 90.0


def _make_tools(calls: list[ToolCallRecord], max_steps: int):
    """Builds the 5 AG2-decorated tools for one run, each recording a
    ToolCallRecord into `calls` as a side effect - same closure pattern as
    the other two adapters, plus the step-budget circuit breaker described
    in the module docstring."""

    def _record(name: str, args: dict, fn, *fn_args):
        if len(calls) >= max_steps and name not in ("answer", "escalate"):
            error = f"{name}: step budget ({max_steps}) exhausted - call answer() or escalate() now"
            calls.append(ToolCallRecord(
                step=len(calls), tool_name=name, tool_args=args, result=None,
                error=error, latency_ms=0.0,
            ))
            return {"error": error}
        start = time.perf_counter()
        try:
            result = fn(*fn_args)
            calls.append(ToolCallRecord(
                step=len(calls), tool_name=name, tool_args=args, result=result,
                error=None, latency_ms=(time.perf_counter() - start) * 1000,
            ))
            return result
        except tool_impl.ToolInputError as exc:
            calls.append(ToolCallRecord(
                step=len(calls), tool_name=name, tool_args=args, result=None,
                error=str(exc), latency_ms=(time.perf_counter() - start) * 1000,
            ))
            return {"error": str(exc)}

    @ag2.tool
    def search_orders(query: str):
        """Free-text search across order id, item, and customer name."""
        return _record("search_orders", {"query": query}, tool_impl.search_orders, query)

    @ag2.tool
    def get_customer(customer_id: str):
        """Look up one customer by id, format CUST-####."""
        return _record(
            "get_customer", {"customer_id": customer_id}, tool_impl.get_customer, customer_id
        )

    @ag2.tool
    def calculate_refund(order_id: str, reason: str):
        """Compute refund eligibility and amount for one order. reason must be one of:
        defective, not_as_described, changed_mind, late_delivery, fraud_suspected."""
        return _record(
            "calculate_refund", {"order_id": order_id, "reason": reason},
            tool_impl.calculate_refund, order_id, reason,
        )

    @ag2.tool
    def escalate(reason: str):
        """Terminal action: hand this off to a human instead of resolving it yourself."""
        return _record("escalate", {"reason": reason}, tool_impl.escalate, reason)

    @ag2.tool
    def answer(text: str):
        """Terminal action: your final response to the user's question."""
        return _record("answer", {"text": text}, tool_impl.answer, text)

    return [search_orders, get_customer, calculate_refund, escalate, answer]


async def _run_async(
    task: GoldenTask, max_steps: int, model: str, calls: list[ToolCallRecord]
) -> None:
    agent = ag2.Agent(
        name="support_ops_assistant",
        prompt=SHARED_SYSTEM_PROMPT.format(max_steps=max_steps),
        config=AnthropicConfig(model=model, max_tokens=1024),
        tools=_make_tools(calls, max_steps),
    )
    await agent.ask(task.prompt)


def run_one(task: GoldenTask, trial_idx: int, max_steps: int, model: str) -> Trajectory:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_start = time.perf_counter()
    calls: list[ToolCallRecord] = []

    traj = Trajectory(
        framework="ag2", task_id=task.task_id, trial_idx=trial_idx, model=model,
        started_at=started_at,
    )

    try:
        asyncio.run(asyncio.wait_for(
            _run_async(task, max_steps, model, calls), timeout=_WALL_CLOCK_TIMEOUT_S
        ))
        terminal_action, record = determine_terminal(calls, max_steps)
        traj.terminal_action = terminal_action
        if terminal_action == "answer":
            traj.final_text = extract_answer_text(record)
    except asyncio.TimeoutError:
        traj.terminal_action = "max_steps"
    except Exception as exc:  # noqa: BLE001 - any framework/API failure becomes a graded trial, not a crash
        traj.terminal_action = "fatal_error"
        traj.raw_error = str(exc)

    traj.tool_calls = calls
    traj.step_count = len(calls)
    traj.wall_clock_s = time.perf_counter() - wall_start
    return traj
