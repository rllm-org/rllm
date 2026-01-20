"""
VerifiersWorkflow: rLLM Workflow that uses verifiers for RL environments.

This workflow wraps the rllm RolloutEngine as an AsyncOpenAI-compatible client,
allowing seamless integration with verifiers environments during training.
"""

from typing import Any

import verifiers as vf
from verifiers.types import State, TrajectoryStep
from verifiers.utils.async_utils import maybe_semaphore

from examples.verifiers_env.openai_wrapper import RolloutEngineAsyncClient
from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.workflows.workflow import Workflow


class VerifiersWorkflow(Workflow):
    """
    rLLM Workflow that uses verifiers for RL environments and generating rollouts.

    This workflow:
    1. Wraps the rllm RolloutEngine as an AsyncOpenAI-compatible client
    2. Passes the client to a verifiers Environment for rollout generation
    3. Converts verifiers State/TrajectorySteps to rllm Episode/Trajectory/Step

    Example usage:
        trainer = AgentTrainer(
            workflow_class=VerifiersWorkflow,
            workflow_args={
                "vf_env": vf.load_environment("math", ...),
            },
            train_dataset=train_dataset,
            config=config,
        )
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        vf_env: vf.Environment,
        sampling_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, **kwargs)
        self.vf_env = vf_env
        self.sampling_args = sampling_args or {}

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Run a single rollout using the verifiers environment.

        Args:
            task: Task dict containing 'prompt', 'example_id', 'task', etc.
            uid: Unique identifier for this rollout (format: "task_id:rollout_idx")

        Returns:
            Episode containing the trajectory with rewards from verifiers
        """
        self.reset(task, uid)

        # Create verifiers-compatible client from rllm's rollout engine
        # TinkerEngine uses model_name, OpenAIEngine uses model
        model = getattr(self.rollout_engine, "model", None) or getattr(self.rollout_engine, "model_name", "unknown")
        client = RolloutEngineAsyncClient(
            rollout_engine=self.rollout_engine,
            model=model,
            application_id_fn=lambda: uid,
        )

        # Build verifiers RolloutInput from rllm task
        rollout_input = {
            "prompt": task.get("prompt") or task.get("messages", []),
            "example_id": task.get("example_id", 0),
            "task": task.get("task", "default"),
        }
        if "answer" in task:
            rollout_input["answer"] = task["answer"]
        if "info" in task:
            rollout_input["info"] = task["info"]

        # Run verifiers rollout
        state: State = await self.vf_env.rollout(
            input=rollout_input,
            client=client,
            model=client._model,
            sampling_args=self.sampling_args,
        )
        score_sem = await maybe_semaphore(-1)  # -1 means no limit

        await self.vf_env.rubric.score_rollout(state, score_sem=score_sem)

        # Convert verifiers State to rllm Episode
        episode = self._convert_state_to_episode(state, uid)

        return episode

    def _convert_state_to_episode(self, state: State, uid: str) -> Episode:
        """
        Convert a verifiers State to an rllm Episode.

        Args:
            state: Completed verifiers State with trajectory and reward
            uid: Unique identifier for this episode

        Returns:
            rllm Episode with converted trajectories
        """
        trajectory_steps: list[TrajectoryStep] = state.get("trajectory", [])

        # Build rllm Steps from verifiers TrajectorySteps
        steps = []
        for traj_step in trajectory_steps:
            tokens = traj_step.get("tokens")
            step = Step(
                prompt_ids=tokens["prompt_ids"] if tokens else [],
                response_ids=tokens["completion_ids"] if tokens else [],
                logprobs=tokens["completion_logprobs"] if tokens else [],
                reward=traj_step.get("reward", 0.0),
            )
            steps.append(step)

        # Create trajectory with final reward
        trajectory = Trajectory(
            steps=steps,
            reward=state.get("reward", 0.0),
            name="verifiers",
        )

        # Create episode
        episode = Episode(
            trajectories=[trajectory],
            is_correct=state.get("reward", 0.0) > 0,
        )
        episode.id = uid
        episode.task = state.get("task", {})

        # Add metrics if available
        if state.get("metrics"):
            episode.metrics = state["metrics"]

        return episode
