"""Offline, deterministic - exercises harness/runner.py's resume logic
against a fake run_agent, so this suite never needs an API key, a real
framework, or a network call."""
import json

from agents.common import Trajectory
from harness import runner
from tasks.schema import GoldenTask


def _task(task_id: str) -> GoldenTask:
    return GoldenTask(
        task_id=task_id, category="lookup", prompt="p", expected_action="answer",
        match_type="contains_all", required_substrings=["ok"], notes="n",
    )


def _fake_run_agent(call_log):
    def _run(framework, task, trial_idx, **kwargs):
        call_log.append((framework, task.task_id, trial_idx))
        return Trajectory(
            framework=framework, task_id=task.task_id, trial_idx=trial_idx, model="fake",
            terminal_action="answer", final_text="ok",
        )
    return _run


def test_first_run_executes_every_combination(tmp_path, monkeypatch):
    call_log: list[tuple[str, str, int]] = []
    monkeypatch.setattr(runner, "run_agent", _fake_run_agent(call_log))
    results_path = tmp_path / "trials.jsonl"

    tasks = [_task("t1"), _task("t2")]
    runner.run(["langgraph"], tasks, trials=2, results_path=results_path)

    assert len(call_log) == 4  # 1 framework x 2 tasks x 2 trials
    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4


def test_second_run_skips_everything_already_recorded(tmp_path, monkeypatch):
    call_log: list[tuple[str, str, int]] = []
    monkeypatch.setattr(runner, "run_agent", _fake_run_agent(call_log))
    results_path = tmp_path / "trials.jsonl"
    tasks = [_task("t1"), _task("t2")]

    runner.run(["langgraph"], tasks, trials=2, results_path=results_path)
    call_log.clear()
    runner.run(["langgraph"], tasks, trials=2, results_path=results_path)

    assert call_log == []  # nothing new should have been executed
    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # file wasn't duplicated either


def test_partial_resume_only_runs_the_new_task(tmp_path, monkeypatch):
    call_log: list[tuple[str, str, int]] = []
    monkeypatch.setattr(runner, "run_agent", _fake_run_agent(call_log))
    results_path = tmp_path / "trials.jsonl"

    runner.run(["langgraph"], [_task("t1")], trials=2, results_path=results_path)
    call_log.clear()
    runner.run(["langgraph"], [_task("t1"), _task("t2")], trials=2, results_path=results_path)

    assert call_log == [("langgraph", "t2", 0), ("langgraph", "t2", 1)]
    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4


def test_each_line_is_valid_json_with_expected_fields(tmp_path, monkeypatch):
    call_log: list[tuple[str, str, int]] = []
    monkeypatch.setattr(runner, "run_agent", _fake_run_agent(call_log))
    results_path = tmp_path / "trials.jsonl"

    runner.run(["langgraph"], [_task("t1")], trials=1, results_path=results_path)

    record = json.loads(results_path.read_text(encoding="utf-8").splitlines()[0])
    for field in ("framework", "task_id", "category", "trial_idx", "pass", "failure_category"):
        assert field in record
    assert record["pass"] is True
    assert record["failure_category"] == "success"
