"""Offline, deterministic - computes metrics from hand-built trial lists, so
this suite never needs an API key, a real framework, or a network call."""
from harness.metrics import framework_summary, per_task_table, write_summary


def _trial(framework, task_id, trial_idx, passed, category="lookup", failure_category=None):
    default_category = "success" if passed else "silent_wrong_answer"
    return {
        "framework": framework, "task_id": task_id, "trial_idx": trial_idx, "pass": passed,
        "category": category, "failure_category": failure_category or default_category,
    }


def test_per_task_table_all_pass():
    trials = [_trial("langgraph", "t1", i, True) for i in range(5)]
    table = per_task_table(trials)
    entry = table[("langgraph", "t1")]
    assert entry["pass_at_1"] is True
    assert entry["pass_at_5"] is True
    assert entry["trial_pattern"] == [True, True, True, True, True]


def test_per_task_table_first_trial_passes_but_not_all():
    trials = [_trial("langgraph", "t1", i, i == 0) for i in range(5)]
    table = per_task_table(trials)
    entry = table[("langgraph", "t1")]
    assert entry["pass_at_1"] is True
    assert entry["pass_at_5"] is False


def test_per_task_table_first_trial_fails():
    trials = [_trial("langgraph", "t1", i, i != 0) for i in range(5)]
    table = per_task_table(trials)
    entry = table[("langgraph", "t1")]
    assert entry["pass_at_1"] is False
    assert entry["pass_at_5"] is False


def test_framework_summary_aggregates_across_tasks():
    trials = [
        *[_trial("langgraph", "t1", i, True) for i in range(5)],  # pass@1 and pass^5
        *[_trial("langgraph", "t2", i, i == 0) for i in range(5)],  # pass@1 only
    ]
    summary = framework_summary(trials)
    stats = summary["langgraph"]
    assert stats["total_tasks"] == 2
    assert stats["total_trials"] == 10
    assert stats["pass_at_1"] == 1.0  # both tasks' trial 0 passed
    assert stats["pass_at_5"] == 0.5  # only t1 passed all 5


def test_framework_summary_counts_taxonomy_categories():
    trials = [
        _trial("crewai", "t1", 0, True, failure_category="success"),
        _trial("crewai", "t1", 1, False, failure_category="tool_call_error"),
        _trial("crewai", "t1", 2, False, failure_category="tool_call_error"),
    ]
    summary = framework_summary(trials)
    assert summary["crewai"]["taxonomy_counts"] == {"success": 1, "tool_call_error": 2}


def test_framework_summary_keeps_frameworks_independent():
    trials = [
        *[_trial("langgraph", "t1", i, True) for i in range(5)],
        *[_trial("crewai", "t1", i, False) for i in range(5)],
    ]
    summary = framework_summary(trials)
    assert summary["langgraph"]["pass_at_5"] == 1.0
    assert summary["crewai"]["pass_at_5"] == 0.0


def test_write_summary_round_trips_through_a_file(tmp_path):
    trials_path = tmp_path / "trials.jsonl"
    out_path = tmp_path / "summary.json"
    lines = [
        '{"framework": "langgraph", "task_id": "t1", "trial_idx": 0, "pass": true, '
        '"category": "lookup", "failure_category": "success"}',
    ]
    trials_path.write_text("\n".join(lines), encoding="utf-8")

    summary = write_summary(trials_path, out_path)

    assert out_path.exists()
    assert summary["total_trials_recorded"] == 1
    assert summary["frameworks"]["langgraph"]["pass_at_1"] == 1.0


def test_write_summary_handles_an_empty_or_missing_file(tmp_path):
    summary = write_summary(tmp_path / "does_not_exist.jsonl", tmp_path / "summary.json")
    assert summary["total_trials_recorded"] == 0
    assert summary["frameworks"] == {}
    assert summary["tasks"] == []
