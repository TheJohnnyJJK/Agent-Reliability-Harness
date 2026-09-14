"""LangGraph adapter: a hand-built StateGraph, not the deprecated
create_react_agent convenience wrapper.

Mirrors lead-router's own agent/graph.py in spirit - a real graph with an
agent node, a tool node, and a conditional edge, not a black-box "just
give it tools and hope" call. The graph itself only decides *when to stop*
(has a terminal tool - answer/escalate - been called, or has the step
budget run out); everything the Trajectory needs (which tools were called,
with what arguments, what came back, how long each took) is captured by the
tool wrappers below as a side effect, then assembled into a Trajectory after
the graph finishes. That keeps the graph's own logic simple and keeps
telemetry capture identical in spirit to the other two adapters.
"""
from __future__ import annotations

import json
import time
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.common import (
    SHARED_SYSTEM_PROMPT,
    ToolCallRecord,
    Trajectory,
    determine_terminal,
    extract_answer_text,
)
from tasks.schema import GoldenTask
from tools import definitions as tool_impl


class _AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _make_tools(calls: list[ToolCallRecord]):
    """Builds the 5 LangChain-decorated tools for one run, each recording a
    ToolCallRecord into `calls` as a side effect. A fresh set is built per
    trial so `calls` never leaks between trials."""

    def _record(name: str, args: dict, fn, *fn_args) -> str:
        start = time.perf_counter()
        try:
            result = fn(*fn_args)
            calls.append(ToolCallRecord(
                step=len(calls), tool_name=name, tool_args=args, result=result,
                error=None, latency_ms=(time.perf_counter() - start) * 1000,
            ))
            return json.dumps(result)
        except tool_impl.ToolInputError as exc:
            calls.append(ToolCallRecord(
                step=len(calls), tool_name=name, tool_args=args, result=None,
                error=str(exc), latency_ms=(time.perf_counter() - start) * 1000,
            ))
            return json.dumps({"error": str(exc)})

    @tool
    def search_orders(query: str) -> str:
        """Free-text search across order id, item, and customer name."""
        return _record("search_orders", {"query": query}, tool_impl.search_orders, query)

    @tool
    def get_customer(customer_id: str) -> str:
        """Look up one customer by id, format CUST-####."""
        return _record(
            "get_customer", {"customer_id": customer_id}, tool_impl.get_customer, customer_id
        )

    @tool
    def calculate_refund(order_id: str, reason: str) -> str:
        """Compute refund eligibility and amount for one order. reason must be one of:
        defective, not_as_described, changed_mind, late_delivery, fraud_suspected."""
        return _record(
            "calculate_refund", {"order_id": order_id, "reason": reason},
            tool_impl.calculate_refund, order_id, reason,
        )

    @tool
    def escalate(reason: str) -> str:
        """Terminal action: hand this off to a human instead of resolving it yourself."""
        return _record("escalate", {"reason": reason}, tool_impl.escalate, reason)

    @tool
    def answer(text: str) -> str:
        """Terminal action: your final response to the user's question."""
        return _record("answer", {"text": text}, tool_impl.answer, text)

    return [search_orders, get_customer, calculate_refund, escalate, answer]


def run_one(task: GoldenTask, trial_idx: int, max_steps: int, model: str) -> Trajectory:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_start = time.perf_counter()
    calls: list[ToolCallRecord] = []

    traj = Trajectory(
        framework="langgraph", task_id=task.task_id, trial_idx=trial_idx, model=model,
        started_at=started_at,
    )

    try:
        tools_list = _make_tools(calls)
        llm = ChatAnthropic(model=model, max_tokens=1024).bind_tools(tools_list)

        def call_model(state: _AgentState) -> dict:
            return {"messages": [llm.invoke(state["messages"])]}

        def route_after_agent(state: _AgentState) -> str:
            last = state["messages"][-1]
            return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

        def route_after_tools(state: _AgentState) -> str:
            called_names: set[str] = set()
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    called_names = {tc["name"] for tc in msg.tool_calls}
                    break
            if called_names & {"answer", "escalate"}:
                return END
            if len(calls) >= max_steps:
                return END
            return "agent"

        graph = StateGraph(_AgentState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(tools_list))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
        graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
        app = graph.compile()

        system = SystemMessage(content=SHARED_SYSTEM_PROMPT.format(max_steps=max_steps))
        app.invoke(
            {"messages": [system, HumanMessage(content=task.prompt)]},
            config={"recursion_limit": max_steps * 2 + 6},
        )

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
