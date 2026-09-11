"""Isolation: CyberGym must not use in-sandbox test.sh."""

from __future__ import annotations

from pathlib import Path

import pytest

from rllm.integrations.harbor.cybergym.eval_guard import (
    CYBERGYM_ISOLATION_ERROR,
    prepare_local_harbor_eval,
    task_is_cybergym,
    tasks_are_cybergym,
)
from rllm.integrations.harbor.utils import is_harbor_agent, resolve_harbor_task_path
from rllm.types import Task

from .cybergym_task_factory import write_cybergym_task


def test_is_harbor_agent_includes_cybergym():
    assert is_harbor_agent("harbor:claude-code") is True
    assert is_harbor_agent("cybergym:openhands") is True
    assert is_harbor_agent("claude-code") is False


def test_resolve_task_path_falls_back_to_task_dir(tmp_path):
    task = Task(id="t", instruction="x", metadata={}, dataset_dir=tmp_path, sub_dir=None)
    assert resolve_harbor_task_path(task) == str(tmp_path)


def test_resolve_task_path_prefers_metadata():
    task = Task(id="t", instruction="x", metadata={"task_path": "/harbor/task"}, dataset_dir=Path("."))
    assert resolve_harbor_task_path(task) == "/harbor/task"


def test_prepare_rejects_in_sandbox_agent(tmp_path):
    task_dir = write_cybergym_task(tmp_path)
    task = Task(id="t", instruction="x", metadata={}, dataset_dir=task_dir)
    assert task_is_cybergym(task) is True
    with pytest.raises(ValueError, match="compose sidecar"):
        prepare_local_harbor_eval("claude-code", [task], None)
    with pytest.raises(ValueError, match="AUTH_TOKEN"):
        prepare_local_harbor_eval("opencode", [task], None)
    assert CYBERGYM_ISOLATION_ERROR


def test_prepare_upgrades_harbor_agent_and_evaluator(tmp_path):
    task_dir = write_cybergym_task(tmp_path)
    task = Task(id="t", instruction="x", metadata={}, dataset_dir=task_dir)
    agent, ev = prepare_local_harbor_eval("harbor:claude-code", [task], None)
    assert agent == "cybergym:claude-code"
    assert ev == "cybergym_reward_fn"
    assert task.metadata["task_path"] == str(task.task_dir)


def test_prepare_keeps_explicit_evaluator(tmp_path):
    task_dir = write_cybergym_task(tmp_path)
    task = Task(id="t", instruction="x", metadata={"task_path": str(task_dir)}, dataset_dir=task_dir)
    agent, ev = prepare_local_harbor_eval("cybergym:openhands", [task], "harbor_reward_fn")
    assert agent == "cybergym:openhands"
    assert ev is None


def test_non_cybergym_harbor_uses_harbor_evaluator(tmp_path):
    (tmp_path / "task.toml").write_text("[task]\nname='swe'\n")
    task = Task(id="t", instruction="x", metadata={"task_path": str(tmp_path)}, dataset_dir=tmp_path)
    assert tasks_are_cybergym([task]) is False
    agent, ev = prepare_local_harbor_eval("harbor:mini-swe-agent", [task], None)
    assert agent == "harbor:mini-swe-agent"
    assert ev == "harbor_reward_fn"
