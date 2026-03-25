"""Compatibility adapter replacing AgentExecutionEngine with workflow execution.

This module provides a light-weight migration path for scripts that currently
instantiate ``rllm.engine.agent_execution_engine.AgentExecutionEngine`` for
inference/evaluation. It preserves the familiar constructor shape while routing
execution through ``UnifiedWorkflowEngine`` + ``CumulativeWorkflow``.
"""

from __future__ import annotations

import logging
from typing import Any

from rllm.agents.agent import Trajectory
from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows.cumulative_workflow import CumulativeWorkflow

logger = logging.getLogger(__name__)


class AgentExecutionWorkflowFactory:
    """Build workflow class/args from legacy agent/env constructor arguments."""

    def __init__(
        self,
        agent_class,
        env_class,
        agent_args: dict | None = None,
        env_args: dict | None = None,
        max_steps: int = 5,
    ) -> None:
        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args or {}
        self.env_args = env_args or {}
        self.max_steps = max_steps

    def get_workflow_cls(self):
        return CumulativeWorkflow

    def get_workflow_args(self) -> dict[str, Any]:
        return {
            "agent_cls": self.agent_class,
            "env_cls": self.env_class,
            "agent_args": self.agent_args,
            "env_args": self.env_args,
            "max_steps": self.max_steps,
        }


class AgentExecutionWorkflowEngine:
    """Workflow-backed compatibility engine for legacy rollout scripts.

    Notes:
    - This class intentionally targets the ``execute_tasks(tasks)`` inference
      interface used by examples like FrozenLake.
    - Returned values are flattened trajectories to remain compatible with
      helpers such as ``compute_pass_at_k`` that expect ``list[Trajectory]``.
    """

    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=128,
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,
        enforce_max_prompt_length=False,
        overlong_filter=False,
        **kwargs,
    ):
        del trajectory_timeout, gamma, max_workers, enforce_max_prompt_length, overlong_filter
        self.config = config
        self.engine_name = engine_name
        self.tokenizer = tokenizer
        self.n_parallel_agents = n_parallel_agents
        self.retry_limit = retry_limit
        self.disable_thinking = self.config.get("rllm", {}).get("disable_thinking", False) if self.config is not None else False

        if agent_args is None:
            agent_args = {}
        if env_args is None:
            env_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}

        # Keep legacy naming for example compatibility.
        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})

        if self.engine_name == "openai":
            from rllm.engine.rollout.openai_engine import OpenAIEngine

            self.rollout_engine = OpenAIEngine(
                **rollout_engine_args,
                api_retries=api_retries,
                tokenizer=self.tokenizer,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                disable_thinking=kwargs.get("disable_thinking", False),
            )
        elif self.engine_name == "verl":
            from rllm.engine.rollout.verl_engine import VerlEngine

            self.rollout_engine = VerlEngine(
                config=self.config,
                rollout_manager=rollout_engine,
                tokenizer=self.tokenizer,
                disable_thinking=kwargs.get("disable_thinking", False),
            )
        elif self.engine_name == "tinker":
            from rllm.engine.rollout.tinker_engine import TinkerEngine

            # CumulativeWorkflow relies on rollout_engine.chat_parser for prompt length accounting.
            # Tinker defaults to renderer mode where chat_parser is None, so we enable parser mode by default.
            rollout_engine_args.setdefault("bypass_render_with_parser", True)
            rollout_engine_args.setdefault("disable_thinking", self.disable_thinking)
            self.rollout_engine = TinkerEngine(
                **rollout_engine_args,
            )
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

        # Backward-compatible parser initialization if caller provides/needs one.
        if chat_parser is None and getattr(self.rollout_engine, "chat_parser", None) is None:
            self.rollout_engine.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=self.disable_thinking)
        elif chat_parser is not None:
            self.rollout_engine.chat_parser = chat_parser

        factory = AgentExecutionWorkflowFactory(
            agent_class=agent_class,
            env_class=env_class,
            agent_args=agent_args,
            env_args=env_args,
            max_steps=max_steps,
        )
        self.workflow_engine = UnifiedWorkflowEngine(
            workflow_cls=factory.get_workflow_cls(),
            workflow_args=factory.get_workflow_args(),
            rollout_engine=self.rollout_engine,
            config=self.config,
            n_parallel_tasks=self.n_parallel_agents,
            retry_limit=self.retry_limit,
            raise_on_error=True,
        )

    async def execute_tasks(self, tasks: list[dict]):
        episodes = await self.workflow_engine.execute_tasks(tasks)
        trajectories: list[Trajectory] = []
        for episode in episodes:
            if not episode.trajectories:
                continue
            for traj in episode.trajectories:
                # Preserve old helper compatibility (expects trajectory.task / reward).
                if traj.task is None:
                    traj.task = episode.task
                trajectories.append(traj)
        return trajectories

    def shutdown(self):
        self.workflow_engine.shutdown()
