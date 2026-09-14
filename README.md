# Agent Reliability Harness

Project #02 from the [Proof of Work](https://claude.ai/code/artifact/02b489c1-763f-4138-bbd0-020e7d0a544e) portfolio, a companion to project #01, [Lead Router](https://github.com/TheJohnnyJJK/Lead-Router). The idea here: prove an agent actually works with numbers and trials, not a demo GIF.

This one builds the same small support-ops agent three times, once each in **LangGraph**, **CrewAI**, and **AG2** (the actively maintained AutoGen fork). Same tools, same dataset, same model, same 22 hand-written test tasks. The only thing that changes between the three is the orchestration framework, so any difference in the results is actually about the framework, not about which model answered the question.

Every task gets run 5 times per framework. Two numbers come out the other end: **pass@1** (would a single run have worked) and **pass^5** (did every one of the 5 runs work). The gap between those two numbers is usually the more interesting one — it's the difference between "it worked when I tried it" and "it's actually reliable."

```
golden task ─▶ same tools, same dataset ─▶ LangGraph agent   ─┐
                                        ├▶ CrewAI agent      ─┼▶ graded, classified ─▶ report
                                        └▶ AG2 agent         ─┘
```

## Why it's built this way

Grading is done with code, not another LLM sitting in judgment. Every golden task has a fixed expected answer or a required action (answer vs. escalate), and `harness/grading.py` checks the agent's output against that mechanically — exact match, a number within tolerance, or a required phrase. That's the same philosophy [Lead Router's eval harness](https://github.com/TheJohnnyJJK/Lead-Router/blob/main/eval/run_eval.py) uses: an eval you can't fully trust the number from isn't much of an eval.

All three agents share one dataset and one set of tools (`search_orders`, `get_customer`, `calculate_refund`, plus two terminal tools, `answer` and `escalate`). That's on purpose. If each framework got its own dataset or its own tool wording, a difference in the results might just mean one setup was easier, not that one framework is actually better.

A failing trial doesn't just get marked "fail." `harness/taxonomy.py` looks at the whole trajectory (every tool call, in order) and sorts it into one of four failure types: the agent looped on the same tool call, a tool call errored and the agent never recovered, the agent invented an ID nobody gave it, or the agent finished cleanly with a wrong answer. Those are different bugs with different fixes, and lumping them into one "fail" count would hide that.

## The golden task set

22 tasks in `tasks/golden_tasks.json`, split into five categories:

- **lookup** (7) — plain questions about an order or customer. The baseline; every framework should pass these every time.
- **refund_calc** (6) — refund math, including two traps: an order past its refund window, and a cancelled order. The refund tool will happily hand back a dollar figure either way — the agent has to notice it isn't actually eligible.
- **escalation_required** (3) — a flagged account or a fraud-suspected reason. These should never get auto-resolved, no matter how clean the numbers look.
- **nonexistent_id** (3) — an order or customer ID that isn't in the dataset. The right move is reporting "not found," not making something up.
- **ambiguous** (3) — requests with no matching tool or policy (a price-match ask, a subscription cancellation, a duplicate-charge dispute). There's nothing to calculate here; the only correct move is escalating.

Every task's `notes` field says what it's actually testing, the same way Lead Router's golden set names its traps instead of hiding them.

## The dataset

A small SQLite database seeded once from `dataset/seed_data.py`, with a fixed random seed and a frozen reference date (`2026-09-15`, not `datetime.now()`) so refund-window math never quietly changes as real time passes. A handful of customers and orders are hand-picked with specific properties the golden tasks rely on (a flagged account, a cancelled order, an order 60 days old); the rest is generated filler for `search_orders` to have something realistic to search across.

No tool in this project ever writes to that database. It's read-only at the tool layer — see Security below.

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                 # add your ANTHROPIC_API_KEY before running any real agent

# Offline - no API key needed, no network calls, this is most of the project:
pytest -q
ruff check .
mypy dataset tools tasks agents harness report

# Cheap smoke test - a couple of real calls against one framework, to confirm the wiring works:
python -m harness.runner --frameworks langgraph --tasks 2 --trials 1

# The real thing - up to 330 paid LLM calls (22 tasks x 3 frameworks x 5 trials).
# It's resumable: interrupt it and re-run the same command, and it picks up
# where it left off instead of re-paying for trials that already finished.
python -m harness.runner
python -m harness.metrics
python -m report.generate_report
# → open report/benchmark_report.html
```

## Per-framework notes

**LangGraph** — a hand-built `StateGraph` with `ToolNode` and `tools_condition`, not the `create_react_agent` convenience wrapper (that's deprecated in favor of `langchain.agents.create_agent`, which is a different abstraction level than the other two frameworks here). This is also the framework built first, since it's closest in spirit to Lead Router's own `agent/graph.py` — a real graph, not a black box.

**CrewAI** — one `Agent`, one `Task`, one `Crew`. Tools go in through the `@tool` decorator from `crewai.tools`, and the shared instructions go into the agent's `backstory` field, since CrewAI has no raw system-message equivalent — that's just how the framework wants instructions handed to it.

**AG2** — this is where the plan changed mid-build. AG2 hit a 1.0 release with a completely redesigned core (`ag2.Agent`, `ag2.tool`) sometime between when this was scoped and when it was actually built, replacing the old `ConversableAgent`-based API most AutoGen writeups still describe. Two things worth knowing if you're touching this code: it now imports as `import ag2`, not `import autogen` (the old alias is gone in 1.0); and `Agent.ask()` has no built-in step-count cap the way LangGraph's graph or CrewAI's `max_iter` do, so `agents/ag2_agent.py` enforces its own — once a trial's step budget is spent, its own tool wrappers start refusing non-terminal tool calls instead of running them, and a 90-second wall-clock timeout backstops that in case a model ignores the refusal.

All three call the same model (`ANTHROPIC_MODEL` in `.env`, `claude-sonnet-5` by default) and use each framework's own standard way of wiring up tools — nobody gets a hand-rolled loop while another gets a convenience wrapper.

## Results

Run the quickstart above and open `report/benchmark_report.html` for the real numbers. This repo ships without a finished run in it on purpose — see Honesty notes below.

## Security

- **Every tool validates its own input.** `tools/definitions.py` raises `ToolInputError` on a malformed order/customer ID or an unrecognized refund reason, before it ever touches the database. An LLM can pass anything; the tool layer doesn't trust it just because it came from a model instead of a form.
- **The dataset is read-only at the tool layer.** There's no write tool anywhere in this project. Whatever an agent does, it can't corrupt its own eval fixture.
- **Every trial has a step budget.** `DEFAULT_MAX_STEPS = 8` in `agents/common.py`, enforced by each framework's own mechanism (LangGraph's graph routing, CrewAI's `max_iter`, AG2's own tool-level circuit breaker) so a broken or looping agent can't run away burning tokens indefinitely.
- **API keys are read from the environment only**, never logged, never written into `results/trials.jsonl` or the generated report.
- **What's intentionally not here:** authentication, rate limiting, anything resembling a production service. This is a benchmark harness that runs on one machine with one person's API key — Lead Router's `agent/security.py` is the project in this portfolio that's actually meant to sit behind a real endpoint.

## Honesty notes

- Every adapter (LangGraph, CrewAI, AG2) was smoke-tested with real API calls against several task categories — plain lookups, the refund-window trap, the flagged-account escalation trap, a nonexistent ID, an ambiguous request — before being called done. That's a handful of real calls per framework, not the full 330-trial run.
- The full benchmark (22 tasks x 3 frameworks x 5 trials) was deliberately **not** run as part of building this. It costs real API money and takes a while, and that run is the actual deliverable a reader should trigger themselves with `python -m harness.runner`, not something baked into the repo ahead of time.
- AG2's current API (`ag2.Agent`/`ag2.tool`) was verified directly against the installed package (`ag2==1.0.5` at the time this was built) rather than trusted from documentation that may describe an older version — AG2's own ecosystem has shifted more than once (`pyautogen` → community `ag2` fork → this 1.0 rewrite), so anyone revisiting this adapter later should re-check the installed version's actual API before assuming the code still matches it.
- The dataviz-skill palette used in the report is the documented default, used as-is — it wasn't re-validated with the color-contrast script here, since none of its values were changed.

## Layout

```
dataset/        the fixed SQLite dataset - schema, seed generator, read-only query helpers
tools/          the 5 shared tools every framework wraps: search_orders, get_customer,
                calculate_refund, escalate, answer
tasks/          golden task schema + golden_tasks.json (22 hand-written test cases)
agents/         common.py (shared contract) + one adapter per framework
harness/        runner.py (resumable trial runner), grading.py, taxonomy.py, metrics.py
report/         generate_report.py - builds the static HTML benchmark report
results/        trials.jsonl (raw per-trial records) + summary.json (computed metrics)
tests/          offline pytest suite - no network calls, no API key needed
pyproject.toml  ruff + mypy config, same rules as Lead Router
```
