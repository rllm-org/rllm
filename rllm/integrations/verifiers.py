"""Verifiers + rLLM integration.

Verifiers is a complete agent harness that handles:
- Dataset loading
- Environment state management
- Rollouts (via env.generate or env.rollout)
- Scoring (via rubric.score_group)

This module bridges Verifiers' output to rLLM's training format.

Usage:
    from verifiers import load_environment
    from rllm.integrations.verifiers import VerifiersIntegration

    # Verifiers handles everything
    env = load_environment("math_group")

    # Integration bridges to rLLM
    integration = VerifiersIntegration(
        verl_engine=verl_engine,
        model_name="Qwen/Qwen2.5-7B",
    )

    # Generate rollouts (Verifiers does the work)
    # Returns Episodes ready for VERL training
    episodes = await integration.generate(env, n_rollouts=8)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout import ModelOutput
from rllm.sdk.proxy.proxy_manager import VerlProxyManager
from rllm.sdk.store.sqlite_store import SqliteTraceStore

if TYPE_CHECKING:
    from verifiers import Environment
    from verifiers.types import State
    from verl import DataProto

logger = logging.getLogger(__name__)


class VerifiersIntegration:
    """Bridge between Verifiers and rLLM training.

    Verifiers handles:
    - Dataset (env.dataset, env.eval_dataset)
    - Rollouts (env.generate, env.rollout)
    - Scoring (env.rubric.score_group)

    This class:
    - Sets up proxy for vLLM → OpenAI endpoint
    - Calls Verifiers' generate()
    - Converts States → Episodes for VERL training
    """

    def __init__(
        self,
        verl_engine: Any,
        model_name: str,
        db_path: str = "~/.rllm/verifiers_traces.db",
        proxy_port: int = 4000,
        proxy_host: str = "127.0.0.1",
    ):
        """Initialize the integration.

        Args:
            verl_engine: VERL engine instance (provides vLLM servers).
            model_name: Model name for proxy config.
            db_path: Path to SQLite trace storage.
            proxy_port: Port for LiteLLM proxy.
            proxy_host: Host for LiteLLM proxy.
        """
        self.model_name = model_name
        self.db_path = db_path
        self._verl_engine = verl_engine

        # Setup proxy (exposes vLLM as OpenAI endpoint)
        self.proxy = VerlProxyManager(
            rollout_engine=verl_engine,
            model_name=model_name,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
        )
        self.proxy.start_proxy_subprocess(
            config=self.proxy.build_proxy_config(),
            db_path=db_path,
            project="verifiers-rllm",
            add_return_token_ids=True,
        )

        # Create OpenAI client for Verifiers
        from openai import AsyncOpenAI

        proxy_url = self.proxy.get_proxy_url(include_v1=True)
        self.client = AsyncOpenAI(base_url=proxy_url, api_key="EMPTY")

        # Trace storage (for token IDs if needed)
        self.store = SqliteTraceStore(db_path=db_path)

    async def generate(
        self,
        env: Environment,
        n_rollouts: int = 8,
        sampling_args: dict | None = None,
        max_concurrent: int = 64,
        use_eval_dataset: bool = False,
    ) -> list[Episode]:
        """Generate rollouts using Verifiers and convert to Episodes.

        Args:
            env: Verifiers Environment (already has dataset internally).
            n_rollouts: Number of rollouts per example (for GRPO).
            sampling_args: Sampling parameters (temperature, max_tokens, etc.).
            max_concurrent: Max concurrent rollouts.
            use_eval_dataset: Use eval_dataset instead of dataset.

        Returns:
            List of Episodes ready for transform_episodes_for_verl().
        """
        sampling_args = sampling_args or {"temperature": 0.7, "max_tokens": 2048}

        # Get dataset from Verifiers environment
        dataset = env.eval_dataset if use_eval_dataset else env.dataset
        if dataset is None:
            raise ValueError("Environment has no dataset")

        logger.info(f"Generating rollouts: {len(dataset)} examples × {n_rollouts} rollouts")

        # Use Verifiers' generate() - it handles everything
        # generate() returns list of States with rewards already computed
        states = await env.generate(
            inputs=dataset,
            client=self.client,
            model=self.model_name,
            sampling_args=sampling_args,
            n=n_rollouts,
            max_concurrent=max_concurrent,
        )

        # Convert Verifiers States → rLLM Episodes
        episodes = self._states_to_episodes(states)
        logger.info(f"Built {len(episodes)} episodes")

        return episodes

    def _states_to_episodes(self, states: list[State]) -> list[Episode]:
        """Convert Verifiers States to rLLM Episodes.

        Each State contains:
        - trajectory: list of turns with tokens
        - reward: final score from rubric
        - completion: full conversation
        - example_id, task, answer, etc.
        """
        episodes = []

        for state in states:
            if state is None:
                continue

            example_id = state.get("example_id", 0)
            rollout_idx = state.get("rollout_idx", 0)
            reward = state.get("reward", 0.0)
            task = state.get("task", "verifiers")

            # Build Steps from Verifiers trajectory
            steps = self._trajectory_to_steps(state)

            # Create Trajectory
            trajectory = Trajectory(
                name=task,
                steps=steps,
                reward=reward if reward is not None else 0.0,
            )

            # Create Episode
            episode = Episode(
                id=f"{example_id}_{rollout_idx}",
                is_correct=(reward or 0.0) >= 1.0,
                trajectories=[trajectory],
                metrics=state.get("metrics", {}),
            )
            episodes.append(episode)

        return episodes

    def _trajectory_to_steps(self, state: State) -> list[Step]:
        """Convert Verifiers trajectory to rLLM Steps."""
        steps = []
        trajectory = state.get("trajectory", [])

        for traj_step in trajectory:
            prompt = traj_step.get("prompt", [])
            completion = traj_step.get("completion", [])
            tokens = traj_step.get("tokens", {})

            # Chat completions = prompt + completion
            chat_completions = list(prompt) + list(completion)

            # Build ModelOutput if token data available
            model_output = None
            if tokens:
                prompt_ids = tokens.get("prompt_ids", [])
                completion_ids = tokens.get("completion_ids", [])
                logprobs = tokens.get("logprobs", [])

                if prompt_ids and completion_ids:
                    content = ""
                    if completion and isinstance(completion, list) and len(completion) > 0:
                        content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])

                    model_output = ModelOutput(
                        text="",
                        content=content,
                        reasoning="",
                        tool_calls=[],
                        prompt_ids=list(prompt_ids),
                        completion_ids=list(completion_ids),
                        logprobs=list(logprobs) if logprobs else [],
                        prompt_length=len(prompt_ids),
                        completion_length=len(completion_ids),
                        finish_reason=traj_step.get("finish_reason", "stop"),
                    )

            step = Step(
                chat_completions=chat_completions,
                model_output=model_output,
                info=traj_step.get("extras", {}),
            )
            steps.append(step)

        return steps

    def shutdown(self) -> None:
        """Cleanup resources."""
        self.proxy.shutdown_proxy()


def transform_episodes_for_verl(
    episodes: list[Episode],
    tokenizer: Any,
    max_prompt_length: int = 2048,
    max_response_length: int = 2048,
) -> DataProto:
    """Transform Episodes to VERL DataProto format.

    Args:
        episodes: Episodes from VerifiersIntegration.generate()
        tokenizer: Tokenizer for padding
        max_prompt_length: Max prompt length
        max_response_length: Max response length

    Returns:
        DataProto ready for VERL training
    """
    from verl import DataProto
    from verl.utils.torch_functional import pad_sequence_to_length

    prompts = []
    responses = []
    rollout_logprobs = []
    step_rewards = []
    episode_ids = []
    trajectory_ids = []
    step_ids = []
    is_correct = []
    traj_mask = []

    for episode in episodes:
        if episode is None:
            continue

        for trajectory in episode.trajectories:
            if not trajectory.steps:
                continue

            trajectory_id = f"{episode.id}_{trajectory.name}"

            for step_idx, step in enumerate(trajectory.steps):
                if step.model_output is None:
                    continue

                prompt_ids = torch.tensor(step.model_output.prompt_ids, dtype=torch.long)

                # Skip overlong prompts
                if len(prompt_ids) > max_prompt_length:
                    logger.warning(f"Skipping step: prompt {len(prompt_ids)} > {max_prompt_length}")
                    continue

                prompts.append(prompt_ids)
                response_ids = torch.tensor(step.model_output.completion_ids, dtype=torch.long)
                responses.append(response_ids)
                traj_mask.append(torch.ones_like(response_ids, dtype=torch.long))

                logprobs = step.model_output.logprobs or []
                if logprobs:
                    rollout_logprobs.append(torch.tensor(logprobs, dtype=torch.float32))
                else:
                    rollout_logprobs.append(torch.zeros(len(response_ids), dtype=torch.float32))

                # Reward on last step only
                is_last = step_idx == len(trajectory.steps) - 1
                step_rewards.append(trajectory.reward if is_last else 0.0)

                step_ids.append(f"{trajectory_id}_{step_idx}")
                trajectory_ids.append(trajectory_id)
                episode_ids.append(episode.id)
                is_correct.append(episode.is_correct)

    if not prompts:
        raise ValueError("No valid steps found in episodes")

    # Pad sequences
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    input_ids = pad_sequence_to_length(prompts, max_prompt_length, pad_id, left_pad=True)
    attention_mask = (input_ids != pad_id).long()
    position_ids = torch.clamp(attention_mask.cumsum(dim=-1) - 1, min=0)
    response_tensors = pad_sequence_to_length(responses, max_response_length, pad_id, left_pad=False)
    response_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
    logprobs_tensor = pad_sequence_to_length(rollout_logprobs, max_response_length, 0.0, left_pad=False)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": response_tensors,
        "response_mask": response_mask,
        "rollout_log_probs": logprobs_tensor,
        "rewards": torch.tensor(step_rewards, dtype=torch.float32),
    }

    non_tensor_batch = {
        "episode_ids": episode_ids,
        "trajectory_ids": trajectory_ids,
        "step_ids": step_ids,
        "is_correct": is_correct,
    }

    return DataProto.from_dict(tensors=batch, non_tensors=non_tensor_batch)
