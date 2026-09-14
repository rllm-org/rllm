"""Harbor task-path resolution and agent-name helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from rllm.integrations.harbor.utils import is_harbor_agent, resolve_harbor_task_path
from rllm.types import Task


def test_is_harbor_agent_prefix():
    assert is_harbor_agent("harbor:claude-code") is True
    assert is_harbor_agent("claude-code") is False
    assert is_harbor_agent(None) is False


def test_resolve_task_path_prefers_metadata():
    task = Task(id="t", instruction="x", metadata={"task_path": "/harbor/task"}, dataset_dir=Path("."))
    assert resolve_harbor_task_path(task) == "/harbor/task"


def test_resolve_task_path_falls_back_to_task_dir(tmp_path):
    task = Task(id="t", instruction="x", metadata={}, dataset_dir=tmp_path, sub_dir=None)
    assert resolve_harbor_task_path(task) == str(tmp_path)


def test_resolve_task_path_from_row_dict():
    assert resolve_harbor_task_path({"task_path": "/from/row"}) == "/from/row"


def test_resolve_task_path_missing_raises():
    task = Task(id="t", instruction="x", metadata={}, dataset_dir=None)
    with pytest.raises(ValueError, match="task_path"):
        resolve_harbor_task_path(task)
