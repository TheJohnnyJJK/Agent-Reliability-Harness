"""Offline, deterministic - grades canned Trajectory fixtures against canned
GoldenTasks, so this suite never needs an API key, a real framework, or a
network call."""
from agents.common import Trajectory
from harness.grading import grade
from tasks.schema import GoldenTask


def _task(**overrides) -> GoldenTask:
    defaults = {
        "task_id": "t1", "category": "lookup", "prompt": "p", "expected_action": "answer",
        "match_type": "contains_all", "required_substrings": ["ok"], "notes": "n",
    }
    return GoldenTask(**{**defaults, **overrides})


def _traj(**overrides) -> Trajectory:
    defaults = {"framework": "langgraph", "task_id": "t1", "trial_idx": 0, "model": "m"}
    return Trajectory(**{**defaults, **overrides})


def test_exact_match_passes():
    task = _task(match_type="exact", expected_answer="Yes, it shipped.")
    traj = _traj(terminal_action="answer", final_text="  yes, it shipped.  ")
    assert grade(task, traj) is True


def test_exact_match_fails_on_different_text():
    task = _task(match_type="exact", expected_answer="Yes.")
    traj = _traj(terminal_action="answer", final_text="No.")
    assert grade(task, traj) is False


def test_numeric_tolerance_passes_within_bound():
    task = _task(match_type="numeric_tolerance", expected_answer="29.99", tolerance=0.01)
    traj = _traj(terminal_action="answer", final_text="The refund is $29.99.")
    assert grade(task, traj) is True


def test_numeric_tolerance_prefers_dollar_number_over_earlier_bare_number():
    # The order was placed 10 days ago, but the refund figure is $29.99 -
    # grading must not grab the "10" first.
    task = _task(match_type="numeric_tolerance", expected_answer="29.99", tolerance=0.01)
    traj = _traj(terminal_action="answer", final_text="Ordered 10 days ago; refund due is $29.99.")
    assert grade(task, traj) is True


def test_numeric_tolerance_fails_outside_bound():
    task = _task(match_type="numeric_tolerance", expected_answer="29.99", tolerance=0.01)
    traj = _traj(terminal_action="answer", final_text="The refund is $25.00.")
    assert grade(task, traj) is False


def test_contains_all_requires_every_substring():
    task = _task(match_type="contains_all", required_substrings=["wireless mouse", "29.99"])
    traj = _traj(terminal_action="answer", final_text="It's a Wireless Mouse that cost $29.99.")
    assert grade(task, traj) is True


def test_contains_all_fails_when_one_substring_missing():
    task = _task(match_type="contains_all", required_substrings=["wireless mouse", "29.99"])
    traj = _traj(terminal_action="answer", final_text="It's a Wireless Mouse.")
    assert grade(task, traj) is False


def test_escalate_expected_grades_purely_on_terminal_action():
    task = _task(expected_action="escalate", match_type="exact")
    traj = _traj(terminal_action="escalate", final_text=None)
    assert grade(task, traj) is True


def test_wrong_terminal_action_fails_regardless_of_text():
    task = _task(expected_action="escalate", match_type="exact")
    traj = _traj(terminal_action="answer", final_text="anything")
    assert grade(task, traj) is False


def test_no_final_text_fails_answer_expected_task():
    task = _task(match_type="contains_all", required_substrings=["ok"])
    traj = _traj(terminal_action="answer", final_text=None)
    assert grade(task, traj) is False
