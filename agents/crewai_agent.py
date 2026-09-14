"""CrewAI adapter: one Agent, one Task, one Crew - the framework's standard
shape for "give it tools and a goal," not a hand-rolled loop.

Like the LangGraph adapter, this never parses CrewAI's own Task output for
grading - it only cares whether our own answer()/escalate() tool was
called, tracked the same way via the closure-captured `calls` list and the
shared agents.common.determine_terminal(). That keeps grading identical
across frameworks no matter how each one narrates its own "final answer."

SHARED_SYSTEM_PROMPT goes into the Agent's `backstory`, not a raw system
message - CrewAI has no equivalent of LangChain's SystemMessage; folding
instructions into role/goal/backstory is the framework's own idiomatic way
to steer an agent, so that's what's used here rather than fighting the
abstraction.
"""
from __future__ import annotations

import time

from crewai import LLM, Agent, Crew, Task
from crewai.tools import tool

from agents.common import (
    SHARED_SYSTEM_PROMPT,
    ToolCallRecord,
    Trajectory,
    determine_terminal,
    extract_answer_text,
)
from tasks.schema import GoldenTask
from tools import definitions as tool_impl


def _make_tools(calls: list[ToolCallRecord]):
    """Builds the 5 CrewAI-decorated tools for one run, each recording a
    ToolCallRecord into `calls` as a side effect - mirrors
    langgraph_agent.py's _make_tools exactly, just with crewai.tools.tool
    instead of langchain_core.tools.tool."""

    def _record(name: str, args: dict, fn, *fn_args):
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

    @tool
    def search_orders(query: str):
        """Free-text search across order id, item, and customer name."""
        return _record("search_orders", {"query": query}, tool_impl.search_orders, query)

    @tool
    def get_customer(customer_id: str):
        """Look up one customer by id, format CUST-####."""
        return _record(
            "get_customer", {"customer_id": customer_id}, tool_impl.get_customer, customer_id
        )

    @tool
    def calculate_refund(order_id: str, reason: str):
        """Compute refund eligibility and amount for one order. reason must be one of:
        defective, not_as_described, changed_mind, late_delivery, fraud_suspected."""
        return _record(
            "calculate_refund", {"order_id": order_id, "reason": reason},
            tool_impl.calculate_refund, order_id, reason,
        )

    @tool
    def escalate(reason: str):
        """Terminal action: hand this off to a human instead of resolving it yourself."""
        return _record("escalate", {"reason": reason}, tool_impl.escalate, reason)

    @tool
    def answer(text: str):
        """Terminal action: your final response to the user's question."""
        return _record("answer", {"text": text}, tool_impl.answer, text)

    return [search_orders, get_customer, calculate_refund, escalate, answer]


def run_one(task: GoldenTask, trial_idx: int, max_steps: int, model: str) -> Trajectory:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_start = time.perf_counter()
    calls: list[ToolCallRecord] = []

    traj = Trajectory(
        framework="crewai", task_id=task.task_id, trial_idx=trial_idx, model=model,
        started_at=started_at,
    )

    try:
        agent = Agent(
            role="Support Ops Assistant",
            goal="Resolve the user's request accurately using only the tools provided.",
            backstory=SHARED_SYSTEM_PROMPT.format(max_steps=max_steps),
            tools=_make_tools(calls),
            llm=LLM(model=f"anthropic/{model}", max_tokens=1024),
            max_iter=max_steps,
            verbose=False,
        )
        crew_task = Task(
            description=task.prompt,
            expected_output="Whatever text was passed to the answer or escalate tool.",
            agent=agent,
        )
        # tracing=False avoids CrewAI's first-run interactive prompt asking
        # whether to enable its (unrelated, cloud-side) execution tracing.
        Crew(agents=[agent], tasks=[crew_task], verbose=False, tracing=False).kickoff()

        terminal_action, record = determine_terminal(calls, max_steps)
        traj.terminal_action = terminal_action
        if terminal_action == "answer":
            traj.final_text = extract_answer_text(record)
    except Exception as exc:  # noqa: BLE001 - any framework/API failure becomes a graded trial, not a crash
        traj.terminal_action = "fatal_error"
        traj.raw_error = str(exc)

    traj.tool_calls = calls
    traj.step_count = len(calls)
    traj.wall_clock_s = time.perf_counter() - wall_start
    return traj
