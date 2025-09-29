import asyncio
import logging
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.workflows.workflow import TerminationReason, Workflow

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from verl import DataProto

logger = logging.getLogger(__name__)


class AgentWorkflowEngine:
    def __init__(self, workflow_cls: type[Workflow], workflow_args: dict, rollout_engine: RolloutEngine, config=None, n_parallel_tasks=128, retry_limit=3, **kwargs):
        self.workflow_cls = workflow_cls
        self.workflow_args = workflow_args

        self.rollout_engine = rollout_engine
        self.config = config  # if training

        self.retry_limit = retry_limit  # number of attempts to retry a task
        self.kwargs = kwargs

        self.n_parallel_tasks = n_parallel_tasks
        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel_tasks)

        self.workflow_queue = None

    async def initialize_pool(self):
        """A coroutine to create and populate the workflow queue."""
        if self.workflow_queue is not None:
            return
        self.workflow_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
        for i in range(self.n_parallel_tasks):
            workflow = self.workflow_cls(rollout_engine=self.rollout_engine, executor=self.executor, **self.workflow_args)
            assert workflow.is_multithread_safe(), "Workflows must contain only thread-save environments"
            self.workflow_queue.put_nowait(workflow)

    async def execute_tasks(self, tasks: list[dict], task_ids: list[str] | None = None, workflow_id: str | None = None, **kwargs) -> list[Episode]:
        """
        Run asynchronous workflow with retry logic.

        Args:
            tasks: List of tasks to process
            task_ids: List of task ids (not unique if n_rollouts > 1)
            workflow_id: Optional workflow identifier for grouping episodes in storage
        """
        if self.workflow_queue is None:
            await self.initialize_pool()

        if task_ids is not None:
            # Ensure we generate one episode_id per task. If a single base id is
            # provided (common case), reuse it and suffix with incremental counters.
            counters = defaultdict(int)
            episode_ids = []
            for i in range(len(tasks)):
                base_id = task_ids[0] if len(task_ids) == 1 else task_ids[i % len(task_ids)]
                episode_id = f"{base_id}_{counters[base_id]}"
                episode_ids.append(episode_id)
                counters[base_id] += 1
        else:
            episode_ids = [str(uuid.uuid4()) for _ in tasks]

        async def process_task_with_retry(task: dict, uid: str) -> Episode:
            """Process a single task with retry logic"""
            workflow = await self.workflow_queue.get()
            try:
                for retry_attempt in range(1, self.retry_limit + 1):
                    try:
                        episode = await workflow.run_with_termination_handling(task=task, uid=uid, **kwargs)
                        return episode
                    except Exception as e:
                        logger.warning(f"Rollout {uid} failed on attempt {retry_attempt}/{self.retry_limit}: {e}")
                        if retry_attempt == self.retry_limit:
                            raise Exception(f"Rollout {uid} failed permanently.") from e
                        continue
            finally:
                await self.workflow_queue.put(workflow)

        futures = [process_task_with_retry(task, uid) for task, uid in zip(tasks, episode_ids, strict=False)]

        id_to_position = {episode_id: i for i, episode_id in enumerate(episode_ids)}
        results = [None] * len(tasks)

        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                result = await future
                position = id_to_position[result.id]  # episode.id
                results[position] = result
                pbar.update(1)

        return results

    async def execute_tasks_verl(self, batch: "DataProto", workflow_id: str | None = None, **kwargs) -> "DataProto":
        self.rollout_engine.wake_up()
        if batch.meta_info.get("validate", False):
            self.rollout_engine.validate = True
        tasks = batch.non_tensor_batch["extra_info"].tolist()
        task_ids = batch.non_tensor_batch["task_ids"].tolist()
        results = await self.execute_tasks(tasks, task_ids, workflow_id=workflow_id, **kwargs)  # list of Episodes
        self.rollout_engine.validate = False
        self.rollout_engine.sleep()
        return self._transform_results_for_verl(results, task_ids)

    def _transform_results_for_verl(self, episodes: list[Episode], task_ids: np.ndarray) -> "DataProto":
        # Local import to keep verl optional
        from verl import DataProto  # type: ignore
        from verl.utils.torch_functional import pad_sequence_to_length  # type: ignore

        prompts = []
        responses = []
        traj_rewards = []
        step_rewards = []
        episode_ids = []
        trajectory_ids = []
        step_ids = []
        step_nums = []
        repeat_counts = []
        is_last_step = []
        is_correct = []
        traj_mask = []
        termination_reasons = []
        metrics = []

        for i, episode in enumerate(episodes):
            total_steps = 0

            if all(len(trajectory.steps) == 0 for name, trajectory in episode.trajectories):
                # termination hits before an agent finishes it's first step
                # (e.g., the initial prompt exceeds max_prompt_length or a timeout occurs)
                # we delete the episode from the batch by setting repeat_counts to 0
                logger.info(f"Episode {episode.id} has no valid trajectories, dropping it from the batch")
                repeat_counts.append(0)  # deletes corresponding entry from the batch
                continue

            for name, trajectory in episode.trajectories:
                # name: agent identifier, e.g., solver, critic, etc.

                trajectory_id = f"{task_ids[i]}_{name}"  # unique trajectory identifier e.g., 1234567890_solver

                if len(trajectory.steps) == 0:
                    logger.info(f"Trajectory {trajectory_id} has no steps, skipping")
                    continue

                if not self.config.rllm.stepwise_advantage.enable:
                    if not trajectory.is_cumulative():
                        logger.warning(f"Warning: Trajectory {trajectory_id} is not cumulative, but stepwise mode is not enabled. There could be a token mismatch during trajectory generation.")

                    chat_completions = trajectory.steps[-1].chat_completions

                    prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask(chat_completions, mask_last_assistant_only=False)
                    prompts.append(prompt)
                    responses.append(response)
                    traj_mask.append(mask)
                    step_rewards.append(trajectory.reward)
                    step_ids.append(trajectory_id)

                    n_steps = 1
                else:
                    for step_idx, step in enumerate(trajectory.steps):
                        chat_completions = step.chat_completions
                        prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask(chat_completions, mask_last_assistant_only=True)

                        prompts.append(prompt)
                        responses.append(response)
                        traj_mask.append(mask)

                        step_rewards.append(step.reward)

                        step_id = f"{trajectory_id}_step{step_idx}"  # unique step identifier e.g., 1234567890_solver_step0
                        step_ids.append(step_id)

                    n_steps = len(trajectory.steps)

                trajectory_ids.extend([trajectory_id] * n_steps)
                step_nums.extend([n_steps] * n_steps)
                traj_rewards.extend([trajectory.reward] * n_steps)
                is_last_step.extend([False] * n_steps)
                is_last_step[-1] = True
                total_steps += n_steps

            episode_ids.extend([episode.id] * total_steps)
            is_correct.extend([episode.is_correct] * total_steps)
            termination_reasons.extend([episode.termination_reason if episode.termination_reason is not None else TerminationReason.UNKNOWN] * total_steps)
            metrics.extend([episode.metrics] * total_steps)
            repeat_counts.append(total_steps)

        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in prompts],
            batch_first=True,
            padding_value=self.rollout_engine.tokenizer.pad_token_id,
        ).flip(dims=[1])
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.rollout_engine.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]  # truncate if necessary

        response_batch = torch.nn.utils.rnn.pad_sequence(
            responses,
            batch_first=True,
            padding_value=self.rollout_engine.tokenizer.pad_token_id,
        )
        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.rollout_engine.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]  # truncate if necessary

        input_ids = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(input_ids != self.rollout_engine.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        traj_mask = torch.nn.utils.rnn.pad_sequence(traj_mask, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        traj_mask = traj_mask[:, :max_response_length]  # truncate if necessary

        # Place all rewards to last response token of the last_step response
        traj_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        step_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        valid_response_length_sequences = attention_mask[:, max_prompt_length:].sum(dim=-1)
        for i, (traj_reward, step_reward) in enumerate(zip(traj_rewards, step_rewards, strict=False)):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if last_valid_idx >= 0 and last_valid_idx < traj_rewards_batch.shape[1]:
                traj_rewards_batch[i, last_valid_idx] = traj_reward
                step_rewards_batch[i, last_valid_idx] = step_reward

        # compact filtering
        cf = self.config.rllm.compact_filtering
        is_valid = [True] * len(episode_ids)
        if cf.enable:
            for i in range(len(episode_ids)):
                termination_reason = termination_reasons[i]
                if (cf.mask_max_prompt_length_exceeded and termination_reason == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED) or (cf.mask_max_response_length_exceeded and termination_reason == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED) or (cf.mask_max_turns_exceeded and termination_reason == TerminationReason.MAX_TURNS_EXCEEDED) or (cf.mask_timeout and termination_reason == TerminationReason.TIMEOUT):
                    is_valid[i] = False  # set flag to filter out the episode later (after advantages are computed)

        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts_batch,
                "responses": response_batch,
                "response_mask": traj_mask,
                "traj_rewards": traj_rewards_batch,
                "step_rewards": step_rewards_batch,
            },
            non_tensors={
                "episode_ids": np.array(episode_ids),  # unique identifier for each rollout
                "trajectory_ids": np.array(trajectory_ids),  # unique identifier for each trajectory (shares prefix with task_id) and shared across rollouts
                "step_ids": np.array(step_ids),  # unique identifier for each step (shares prefix with task_id) and shared across rollouts
                "batch_ids": np.array([str(uuid.uuid4())] * len(episode_ids)),  # unique identifier for each batch
                "step_nums": np.array(step_nums),
                "is_correct": np.array(is_correct),
                "termination_reasons": np.array([x.value for x in termination_reasons]),
                "metrics": np.array(metrics),
                "is_valid": np.array(is_valid),
                "is_last_step": np.array(is_last_step),
                "is_pad_step": np.array([False] * len(episode_ids)),
            },
            meta_info={
                "repeat_counts": repeat_counts,
            },
        )

    def shutdown(self):
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
