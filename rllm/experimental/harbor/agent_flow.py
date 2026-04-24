"""Harbor agent flow: wraps Harbor agents as rLLM AgentFlow instances.

Adapts Harbor's Trial pipeline to the AgentFlow protocol so that Harbor
agents (claude-code, mini-swe-agent, etc.) can be used with ``rllm eval``.
"""

from __future__ import annotations

import logging

from rllm.experimental.eval.types import AgentConfig, Task
from rllm.types import Episode

logger = logging.getLogger(__name__)


class HarborAgentFlow:
    """Wraps a Harbor agent as an rLLM AgentFlow for eval.

    Uses Harbor's full trial pipeline (environment setup, agent run,
    verification) and returns the result as an rLLM Episode. The container
    lifecycle is fully managed by Harbor's Trial class.

    The ``arun`` async method is preferred by ``EvalRunner``.
    """

    max_concurrent: int = 4

    def __init__(
        self,
        agent_name: str = "mini-swe-agent",
        environment_type: str | None = None,
        agent_kwargs: dict | None = None,
    ):
        self.agent_name = agent_name
        self.environment_type = environment_type
        self.agent_kwargs = agent_kwargs or {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization: set dummy API keys and silence Harbor logs."""
        if self._initialized:
            return
        from rllm.experimental.harbor.trial_helper import (
            ensure_dummy_api_keys,
            silence_harbor,
        )

        ensure_dummy_api_keys()
        silence_harbor()
        self._initialized = True

    async def arun(self, task: Task, config: AgentConfig) -> Episode:
        """Run a Harbor trial and return the result as an Episode.

        Args:
            task: rLLM Task wrapping a dict with ``task_path`` field.
            config: AgentConfig with ``base_url`` and ``model``.

        Returns:
            Episode with harbor_reward and harbor_is_correct in artifacts.
        """
        from rllm.experimental.harbor.trial_helper import (
            build_harbor_trial_config,
            run_harbor_trial,
            trial_result_to_episode,
            trial_result_to_reward,
        )

        self._ensure_initialized()

        task_path = task.data.get("task_path")
        if not task_path:
            raise ValueError(f"Harbor task missing 'task_path' field in task data: {list(task.data.keys())}")

        trial_config = build_harbor_trial_config(
            task_path=task_path,
            agent_name=self.agent_name,
            model_name=config.model,
            inference_url=config.base_url,
            environment_type=self.environment_type,
            agent_kwargs=self.agent_kwargs,
            trial_name=config.session_uid,
        )

        result = await run_harbor_trial(trial_config)
        episode = trial_result_to_episode(result, config.session_uid, task.data)

        # Store reward in artifacts so HarborEvaluator can read it.
        # Use a sentinel key "harbor_trial_ran" to distinguish "trial ran,
        # no reward" (score 0) from "non-Harbor agent, no trial at all".
        reward, is_correct, _ = trial_result_to_reward(result)
        episode.artifacts["harbor_trial_ran"] = True
        episode.artifacts["harbor_reward"] = reward if reward is not None else 0.0
        episode.artifacts["harbor_is_correct"] = is_correct

        return episode

    def run(self, task: Task, config: AgentConfig) -> Episode:
        """Sync wrapper for arun. Used when no event loop is available."""
        import asyncio

        return asyncio.run(self.arun(task, config))


def default_harbor_agent() -> HarborAgentFlow:
    """Factory function for the default Harbor agent (mini-swe-agent)."""
    return HarborAgentFlow(agent_name="mini-swe-agent")
