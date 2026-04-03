"""Shared dataset-evaluation utilities."""

import ast
import json


def make_log(verbose: bool):
    """Create a logging function gated on verbose flag."""
    if verbose:
        return print
    return lambda *_args, **_kwargs: None


def short_id(instance_id: str) -> str:
    """Derive a short display ID from a full instance_id."""
    if "." in instance_id:
        return instance_id.split(".")[-1][:8]
    if "-" in instance_id:
        return instance_id.split("-")[0]
    return instance_id[:30]


def zero_result(**overrides) -> dict:
    """Return a zero-reward grading result dict with optional overrides."""
    result = {"reward": 0.0, "f2p_passed": 0, "f2p_total": 0, "p2p_passed": 0, "p2p_total": 0}
    result.update(overrides)
    return result


def parse_test_list(value) -> list:
    """Parse test list from list, JSON string, or Python literal string.

    DatasetRegistry may serialize lists as JSON strings,
    while CSV files use Python-style string lists.
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []

    value = value.strip()
    if not value or value == "[]":
        return []

    try:
        result = json.loads(value)
        return list(result) if isinstance(result, (list, set)) else []
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        result = ast.literal_eval(value)
        return list(result) if isinstance(result, (list, set, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def normalize_task_test_lists(task: dict, keys: tuple[str, ...] = ("FAIL_TO_PASS", "PASS_TO_PASS")) -> dict:
    """Ensure test list fields in a task dict are actual lists, not JSON strings.

    Returns a shallow copy of task with normalized fields.
    """
    task = dict(task)
    for key in keys:
        val = task.get(key, [])
        task[key] = parse_test_list(val) if val is not None else []
    return task
