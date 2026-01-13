import asyncio
from typing import Any

import httpx
import hydra
import verifiers as vf
from openai import AsyncOpenAI
from verifiers.types import GenerateOutputs, State, TrajectoryStep

from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.rollout import OpenAIEngine
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class VerifiersWorkflow(Workflow):
    """
    rLLM Workflow that uses verifiers for RL environments and generating rollouts.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        # vf_env: vf.Environment,
        # client: AsyncOpenAI | None = None,
        model_name: str,
        env_args: dict | None = None,
        max_steps: int = 50,
        sampling_args: dict[str, Any] | None = None,
        rollouts_per_example: int = 1,
        max_concurrent: int = 64,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, **kwargs)
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.rollouts_per_example = rollouts_per_example
        self.max_concurrent = max_concurrent

        self._client = client
        self.uid = None
        self.task = None
        self._completed_trajectories = []

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)
        vf_output: GenerateOutputs = await self.vf_env.generate(
            inputs=
        )
