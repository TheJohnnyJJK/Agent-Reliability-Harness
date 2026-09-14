"""Turns results/trials.jsonl into results/summary.json.

pass@1 is trial 0's pass rate - what you'd see from a single run per task,
the number a demo GIF implicitly claims. pass^5 is the fraction of tasks
where every one of the 5 trials passed - the number that actually says
"reliable," per the tau-bench-style pass^k idea this whole project is
built around. The gap between the two is the headline finding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_PATH = ROOT / "results" / "trials.jsonl"
SUMMARY_PATH = ROOT / "results" / "summary.json"


def load_trials(path: Path = RESULTS_PATH) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def per_task_table(trials: list[dict]) -> dict[tuple[str, str], dict]:
    """Keyed by (framework, task_id) -> trial-by-trial pass pattern plus
    pass@1/pass^5 for that one task."""
    table: dict[tuple[str, str], dict] = {}
    for record in trials:
        key = (record["framework"], record["task_id"])
        entry = table.setdefault(key, {
            "framework": record["framework"], "task_id": record["task_id"],
            "category": record["category"], "trials": {},
        })
        entry["trials"][record["trial_idx"]] = record["pass"]
    for entry in table.values():
        highest_idx = max(entry["trials"], default=-1)
        ordered = [entry["trials"].get(i, False) for i in range(highest_idx + 1)]
        entry["trial_pattern"] = ordered
        entry["pass_at_1"] = bool(ordered) and ordered[0]
        entry["pass_at_5"] = bool(ordered) and all(ordered)
        del entry["trials"]
    return table


def framework_summary(trials: list[dict]) -> dict[str, dict]:
    """Keyed by framework -> aggregate pass@1/pass^5 across tasks, plus a
    breakdown of failure_category counts across every individual trial."""
    by_task = per_task_table(trials)
    frameworks = sorted({record["framework"] for record in trials})
    summary: dict[str, dict] = {}
    for framework in frameworks:
        task_entries = [e for e in by_task.values() if e["framework"] == framework]
        total_tasks = len(task_entries)
        pass_at_1 = sum(e["pass_at_1"] for e in task_entries) / total_tasks if total_tasks else 0.0
        pass_at_5 = sum(e["pass_at_5"] for e in task_entries) / total_tasks if total_tasks else 0.0

        taxonomy_counts: dict[str, int] = {}
        framework_trials = [r for r in trials if r["framework"] == framework]
        for record in framework_trials:
            cat = record["failure_category"]
            taxonomy_counts[cat] = taxonomy_counts.get(cat, 0) + 1

        summary[framework] = {
            "total_tasks": total_tasks,
            "total_trials": len(framework_trials),
            "pass_at_1": round(pass_at_1, 4),
            "pass_at_5": round(pass_at_5, 4),
            "taxonomy_counts": taxonomy_counts,
        }
    return summary


def write_summary(trials_path: Path = RESULTS_PATH, out_path: Path = SUMMARY_PATH) -> dict:
    trials = load_trials(trials_path)
    by_task = per_task_table(trials)
    summary = {
        "total_trials_recorded": len(trials),
        "frameworks": framework_summary(trials),
        "tasks": [
            {
                "framework": e["framework"], "task_id": e["task_id"], "category": e["category"],
                "trial_pattern": e["trial_pattern"],
                "pass_at_1": e["pass_at_1"], "pass_at_5": e["pass_at_5"],
            }
            for e in sorted(by_task.values(), key=lambda e: (e["framework"], e["task_id"]))
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = write_summary()
    print(f"wrote {SUMMARY_PATH} from {result['total_trials_recorded']} recorded trials")
    for framework, stats in result["frameworks"].items():
        print(f"  {framework}: pass@1={stats['pass_at_1']:.0%} pass^5={stats['pass_at_5']:.0%} "
              f"({stats['total_tasks']} tasks, {stats['total_trials']} trials)")
