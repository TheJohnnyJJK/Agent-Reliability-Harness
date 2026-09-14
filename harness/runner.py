"""Resumable CLI: runs framework agents against golden tasks and appends
graded results to results/trials.jsonl.

Resumable because a full run (~22 tasks x 3 frameworks x 5 trials) is up to
330 real, paid LLM-backed calls. If it's interrupted - a rate limit, a
network blip, Ctrl-C - restarting should not re-pay for trials that already
finished. Every (framework, task_id, trial_idx) already present in
trials.jsonl is treated as done and skipped; each new trial is written and
flushed immediately, not batched at the end, so a crash mid-run loses at
most the one trial in flight.

Usage:
    python -m harness.runner                                   # full run
    python -m harness.runner --frameworks langgraph --tasks 2 --trials 1   # cheap smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

# Must run before importing agents.common - its DEFAULT_MODEL constant reads
# ANTHROPIC_MODEL from the environment at import time.
load_dotenv(ROOT / ".env")

from agents.common import ALL_FRAMEWORKS, Framework, Trajectory, run_agent  # noqa: E402
from dataset import db  # noqa: E402
from harness.grading import grade  # noqa: E402
from harness.taxonomy import classify  # noqa: E402
from tasks.schema import GoldenTask, load_golden_tasks  # noqa: E402

RESULTS_PATH = ROOT / "results" / "trials.jsonl"
DEFAULT_TRIALS = 5


def _load_done(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        done.add((rec["framework"], rec["task_id"], rec["trial_idx"]))
    return done


def _trajectory_to_record(task: GoldenTask, traj: Trajectory) -> dict:
    return {
        "framework": traj.framework,
        "task_id": task.task_id,
        "category": task.category,
        "trial_idx": traj.trial_idx,
        "pass": grade(task, traj),
        "failure_category": classify(task, traj),
        "terminal_action": traj.terminal_action,
        "final_text": traj.final_text,
        "step_count": traj.step_count,
        "wall_clock_s": traj.wall_clock_s,
        "raw_error": traj.raw_error,
        "model": traj.model,
        "timestamp": traj.started_at,
        "tool_calls": [
            {
                "step": c.step, "tool_name": c.tool_name, "tool_args": c.tool_args,
                "result": c.result, "error": c.error, "latency_ms": c.latency_ms,
            }
            for c in traj.tool_calls
        ],
    }


def run(
    frameworks: list[Framework],
    tasks: list[GoldenTask],
    trials: int,
    results_path: Path = RESULTS_PATH,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(results_path)
    total = len(frameworks) * len(tasks) * trials
    completed = 0
    with results_path.open("a", encoding="utf-8") as f:
        for framework in frameworks:
            for task in tasks:
                for trial_idx in range(trials):
                    completed += 1
                    key = (framework, task.task_id, trial_idx)
                    if key in done:
                        print(f"[{completed}/{total}] skip (already recorded): {key}")
                        continue
                    print(f"[{completed}/{total}] running: {key}")
                    traj = run_agent(framework, task, trial_idx)
                    record = _trajectory_to_record(task, traj)
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    status = "PASS" if record["pass"] else "FAIL"
                    print(f"    -> {status} ({record['failure_category']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run framework agents against golden tasks.")
    parser.add_argument(
        "--frameworks", default=",".join(ALL_FRAMEWORKS),
        help="comma-separated subset of: " + ",".join(ALL_FRAMEWORKS),
    )
    parser.add_argument(
        "--tasks", type=int, default=None,
        help="only run the first N golden tasks (default: all)",
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help=f"trials per task (default: {DEFAULT_TRIALS})",
    )
    parser.add_argument(
        "--reseed", action="store_true",
        help="reseed the dataset before running, even if a database file already exists",
    )
    args = parser.parse_args()

    if args.reseed or not Path(db.DB_PATH).exists():
        db.reset_db()

    frameworks = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    all_tasks = load_golden_tasks()
    tasks = all_tasks[: args.tasks] if args.tasks else all_tasks

    run(frameworks, tasks, args.trials)


if __name__ == "__main__":
    main()
