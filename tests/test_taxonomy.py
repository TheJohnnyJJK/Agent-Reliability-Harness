"""Offline, deterministic - classifies canned Trajectory fixtures, so this
suite never needs an API key, a real framework, or a network call."""
from agents.common import ToolCallRecord, Trajectory
from harness.taxonomy import classify
from tasks.schema import GoldenTask


def _task(**overrides) -> GoldenTask:
    defaults = {
        "task_id": "t1", "category": "lookup",
        "prompt": "What is the status of order ORD-1042?",
        "expected_action": "answer", "match_type": "contains_all",
        "required_substrings": ["delivered"], "notes": "n",
    }
    return GoldenTask(**{**defaults, **overrides})


def _traj(**overrides) -> Trajectory:
    defaults = {
        "framework": "langgraph", "task_id": "t1", "trial_idx": 0, "model": "m", "tool_calls": [],
    }
    return Trajectory(**{**defaults, **overrides})


def _call(tool_name="search_orders", args=None, result=None, error=None, step=0) -> ToolCallRecord:
    return ToolCallRecord(
        step=step, tool_name=tool_name, tool_args=args or {}, result=result, error=error,
        latency_ms=1.0,
    )


def test_success_when_grading_passes():
    task = _task()
    traj = _traj(
        terminal_action="answer", final_text="Order ORD-1042 was delivered.",
        tool_calls=[
            _call("search_orders", {"query": "ORD-1042"}, result={"order_id": "ORD-1042"}),
        ],
    )
    assert classify(task, traj) == "success"


def test_silent_wrong_answer_when_terminal_action_clean_but_grading_fails():
    task = _task()
    traj = _traj(terminal_action="answer", final_text="Order ORD-1042 is still processing.")
    assert classify(task, traj) == "silent_wrong_answer"


def test_max_steps_is_infinite_loop():
    task = _task()
    traj = _traj(terminal_action="max_steps", final_text=None)
    assert classify(task, traj) == "infinite_loop"


def test_repeated_identical_tool_call_is_infinite_loop():
    task = _task()
    calls = [
        _call("get_customer", {"customer_id": "CUST-1001"}, result={"found": True}, step=i)
        for i in range(3)
    ]
    traj = _traj(terminal_action="max_steps", final_text=None, tool_calls=calls)
    assert classify(task, traj) == "infinite_loop"


def test_unrecovered_tool_error_is_tool_call_error():
    task = _task()
    traj = _traj(
        terminal_action=None, final_text=None,
        tool_calls=[_call("calculate_refund", {"order_id": "bad"}, error="ToolInputError: bad id")],
    )
    assert classify(task, traj) == "tool_call_error"


def test_hallucinated_parameter_when_id_not_from_prompt_or_dataset_or_prior_result():
    task = _task(prompt="What is the status of order ORD-1042?")
    traj = _traj(
        terminal_action="answer", final_text="Order ORD-4242 was delivered.",
        tool_calls=[
            _call("search_orders", {"query": "ORD-4242"}, result={"order_id": "ORD-4242"}),
        ],
    )
    assert classify(task, traj) == "hallucinated_parameter"


def test_reusing_prompt_supplied_nonexistent_id_is_not_hallucination():
    # The prompt itself hands the agent a nonexistent id (ORD-9999) - calling
    # a tool with it is correct, not invented. This must classify by the
    # normal success/silent_wrong_answer path, never as hallucinated_parameter.
    task = _task(
        prompt="What is the status of order ORD-9999?",
        required_substrings=["could not find"],
    )
    traj = _traj(
        terminal_action="answer", final_text="I could not find that order.",
        tool_calls=[_call("search_orders", {"query": "ORD-9999"}, result=None, error=None)],
    )
    assert classify(task, traj) == "success"


def test_id_returned_by_an_earlier_tool_call_is_legitimately_known():
    # get_customer's own result surfaces CUST-1001; reusing that id in a
    # later call is legitimate, not hallucinated.
    task = _task(prompt="Look up the customer for order ORD-1042.")
    first_result = {"order_id": "ORD-1042", "customer_id": "CUST-1001"}
    traj = _traj(
        terminal_action="answer", final_text="Order ORD-1042 was delivered.",
        tool_calls=[
            _call("search_orders", {"query": "ORD-1042"}, result=first_result, step=0),
            _call("get_customer", {"customer_id": "CUST-1001"}, result={"found": True}, step=1),
        ],
    )
    assert classify(task, traj) == "success"


def test_id_returned_inside_a_list_shaped_result_is_legitimately_known():
    # search_orders returns a list of dicts (possibly several matches), not
    # a single dict - _ids_in_result must walk into that list, not just a
    # top-level dict, or every id learned only from a search result gets
    # misclassified as hallucinated on its next use.
    task = _task(prompt="Search for orders belonging to CUST-1001.")
    search_result = [
        {"order_id": "ORD-1042", "customer_id": "CUST-1001"},
        {"order_id": "ORD-1055", "customer_id": "CUST-1001"},
    ]
    traj = _traj(
        terminal_action="answer", final_text="Order ORD-1055 is delivered.",
        tool_calls=[
            _call("search_orders", {"query": "CUST-1001"}, result=search_result, step=0),
            _call("search_orders", {"query": "ORD-1055"}, result=search_result, step=1),
        ],
    )
    assert classify(task, traj) != "hallucinated_parameter"


def test_looping_takes_precedence_over_tool_error():
    # Both a 3x-repeated call and a tool error are present - infinite_loop
    # must win, per the documented check order in taxonomy.py.
    task = _task()
    calls = [
        _call("get_customer", {"customer_id": "CUST-1001"}, error="boom", step=i)
        for i in range(3)
    ]
    traj = _traj(terminal_action=None, final_text=None, tool_calls=calls)
    assert classify(task, traj) == "infinite_loop"
