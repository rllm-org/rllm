"""Isolation guards so CyberGym never grades inside the agent container."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rllm.integrations.harbor.cybergym.detect import is_cybergym_task_dir
from rllm.integrations.harbor.utils import is_harbor_agent, resolve_harbor_task_path

CYBERGYM_ISOLATION_ERROR = (
    "Harbor CyberGym tasks require a compose sidecar agent "
    "(--agent cybergym:<scaffold> or --agent harbor:<scaffold>). "
    "Running tests/test.sh in the agent sandbox would leak /cybergym binaries "
    "and AUTH_TOKEN. See docs/cybergym/RLLM-HARBOR-PROTOCOL.md."
)


def task_is_cybergym(task: Any) -> bool:
    """True when *task* (Task or row) points at a CyberGym Level 1 directory."""
    try:
        path = resolve_harbor_task_path(task)
    except ValueError:
        task_dir = getattr(task, "task_dir", None)
        path = str(task_dir) if task_dir is not None else ""
    return bool(path) and is_cybergym_task_dir(path)


def tasks_are_cybergym(tasks: Iterable[Any]) -> bool:
    return any(task_is_cybergym(task) for task in tasks)


def stamp_task_paths(tasks: Iterable[Any]) -> None:
    """Ensure each Task.metadata['task_path'] points at the Harbor task dir."""
    for task in tasks:
        metadata = getattr(task, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        if metadata.get("task_path"):
            continue
        task_dir = getattr(task, "task_dir", None)
        if task_dir is not None:
            metadata["task_path"] = str(Path(task_dir))


def upgrade_harbor_agent_for_cybergym(agent_name: str) -> str:
    """``harbor:claude-code`` → ``cybergym:claude-code`` so artifacts are parsed."""
    if agent_name.startswith("harbor:"):
        return "cybergym:" + agent_name.removeprefix("harbor:")
    return agent_name


def prepare_local_harbor_eval(
    agent_name: str | None,
    tasks: list[Any],
    evaluator_name: str | None,
) -> tuple[str | None, str | None]:
    """Stamp paths, enforce isolation, and pick a host-side evaluator.

    Returns ``(agent_name, evaluator_override)``. *evaluator_override* is a
    registry name to use when the caller did not pass ``--evaluator``.
    """
    stamp_task_paths(tasks)
    cybergym = tasks_are_cybergym(tasks)

    if cybergym and not is_harbor_agent(agent_name):
        raise ValueError(CYBERGYM_ISOLATION_ERROR)

    if cybergym and agent_name:
        agent_name = upgrade_harbor_agent_for_cybergym(agent_name)

    if evaluator_name:
        return agent_name, None
    if cybergym:
        return agent_name, "cybergym_reward_fn"
    if is_harbor_agent(agent_name):
        return agent_name, "harbor_reward_fn"
    return agent_name, None
