"""Compatibility adapter for migrating from legacy AgentWorkflowEngine.

This adapter preserves the familiar workflow-engine constructor and runtime
methods while delegating execution to ``UnifiedWorkflowEngine``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine

if TYPE_CHECKING:
    from rllm.agents.agent import Episode
    from rllm.workflows.workflow import Workflow


class AgentWorkflowEngine:
    """Compatibility facade over UnifiedWorkflowEngine.

    The goal is migration ergonomics: old scripts can switch import paths to
    this experimental module with minimal code changes.
    """

    def __init__(
        self,
        workflow_cls: type[Workflow],
        workflow_args: dict,
        rollout_engine,
        config=None,
        n_parallel_tasks: int = 128,
        retry_limit: int = 3,
        raise_on_error: bool = True,
        episode_logger=None,
        **kwargs,
    ):
        self._engine = UnifiedWorkflowEngine(
            workflow_cls=workflow_cls,
            workflow_args=workflow_args,
            rollout_engine=rollout_engine,
            config=config,
            n_parallel_tasks=n_parallel_tasks,
            retry_limit=retry_limit,
            raise_on_error=raise_on_error,
            episode_logger=episode_logger,
            **kwargs,
        )

    @property
    def rollout_engine(self):
        return self._engine.rollout_engine

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0):
        self._engine.set_training_step(step=step, mode=mode, epoch=epoch)

    async def initialize_pool(self):
        await self._engine.initialize_pool()

    async def execute_tasks(self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs) -> list[Episode]:
        return await self._engine.execute_tasks(tasks=tasks, task_ids=task_ids, **kwargs)

    async def execute_tasks_verl(self, batch, **kwargs):
        # Unified workflow path currently returns episodes rather than a
        # transformed DataProto payload.
        return await self._engine.execute_tasks_verl(batch, **kwargs)

    def shutdown(self):
        self._engine.shutdown()
