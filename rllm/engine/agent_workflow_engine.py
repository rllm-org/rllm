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
from rllm.engine.rollout import ModelOutput, RolloutEngine
from rllm.utils import colorful_print
from rllm.workflows.workflow import TerminationReason, Workflow

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from verl import DataProto

logger = logging.getLogger(__name__)


class AgentWorkflowEngine:
    def __init__(self, workflow_cls: type[Workflow], workflow_args: dict, rollout_engine: RolloutEngine, config=None, n_parallel_tasks: int = 128, retry_limit: int = 3, raise_on_error: bool = True, episode_logger=None, **kwargs):
        """Initialize the AgentWorkflowEngine.

        Args:
            workflow_cls: The workflow class to instantiate for each task.
            workflow_args: Arguments to pass to workflow instances.
            rollout_engine: Engine for model inference and rollout.
            config: Optional configuration object for training.
            n_parallel_tasks: Number of parallel workflow instances to maintain.
            retry_limit: Maximum number of retry attempts for failed tasks.
            raise_on_error: Whether to raise exceptions on permanent failures.
            episode_logger: Optional logger for saving episode data to files.
            **kwargs: Additional keyword arguments.
        """
        self.workflow_cls = workflow_cls
        self.workflow_args = workflow_args or {}

        self.rollout_engine = rollout_engine
        self.config = config  # if training

        self.retry_limit = retry_limit  # number of attempts to retry a task
        self.raise_on_error = raise_on_error
        self.kwargs = kwargs

        self.n_parallel_tasks = n_parallel_tasks
        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel_tasks)
        self.workflow_queue = None

        # Episode logging support
        self.episode_logger = episode_logger
        self.current_step = 0
        self.current_epoch = 0
        self.current_mode = "train"  # "train" or "val"

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0):
        """Set current training step for episode logging.

        Args:
            step: Current training step number
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        self.current_step = step
        self.current_mode = mode
        self.current_epoch = epoch

    async def initialize_pool(self):
        """Initialize the workflow pool with parallel workflow instances.

        Creates and populates the workflow queue with workflow instances
        for parallel task processing. This method is idempotent and will
        not recreate the pool if it already exists.
        """
        if self.workflow_queue is not None:
            return
        self.workflow_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
        for i in range(self.n_parallel_tasks):
            workflow = self.workflow_cls(rollout_engine=self.rollout_engine, executor=self.executor, **self.workflow_args)
            assert workflow.is_multithread_safe(), "Workflows must contain only thread-save environments"
            self.workflow_queue.put_nowait(workflow)

    async def process_task_with_retry(self, task: dict, task_id: str, rollout_idx: int, **kwargs) -> tuple[str, int, Episode]:
        """Process a single task rollout with retry logic based on termination reasons.

        Args:
            task: Task dictionary containing the task specification.
            task_id: Unique identifier for the task.
            rollout_idx: Index of this rollout attempt for the task.
            **kwargs: Additional arguments passed to the workflow.

        Returns:
            tuple[str, int, Episode]: Task ID, rollout index, and completed episode.

        Raises:
            Exception: If task fails permanently after retry_limit attempts and raise_on_error is True.
        """
        workflow = await self.workflow_queue.get()
        try:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}"
                episode = await workflow.run_with_termination_handling(task=task, uid=uid, **kwargs)

                # Display rewards for all trajectories
                rewards_str = ", ".join([f"{traj.name}: {traj.reward:.1f}" for traj in episode.trajectories])
                colorful_print(f"[{uid}] Rollout completed. Rewards: {rewards_str}, Termination: {episode.termination_reason}", fg="green" if episode.is_correct else "yellow")

                if episode.termination_reason != TerminationReason.ERROR:
                    return task_id, rollout_idx, episode

                error_tb = episode.info.get("error", {}).get("traceback")
                if error_tb:
                    print(error_tb)

                if retry_attempt < self.retry_limit:
                    print(f"[{uid}] Rollout failed on attempt {retry_attempt}/{self.retry_limit}, retrying...")
                    continue

            if not self.raise_on_error:
                print(f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts.")
            else:
                raise Exception(f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts.")

            return task_id, rollout_idx, episode

        finally:
            await self.workflow_queue.put(workflow)

    async def execute_tasks(self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs) -> list[Episode]:
        """Run asynchronous workflow execution with retry logic for multiple tasks.

        Args:
            tasks: List of task dictionaries to process.
            task_ids: Optional list of task identifiers. If None, UUIDs are generated.
            **kwargs: Additional arguments passed to individual task processing.

        Returns:
            list[Episode]: List of completed episodes from all tasks.
        """
        if self.workflow_queue is None:
            await self.initialize_pool()

        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        task_states = defaultdict(lambda: {"idx": None, "task": None, "episodes": [], "completed": 0, "total_rollouts": 0, "is_complete": False})

        futures = []
        idx_counter = 0
        for task, task_id in zip(tasks, task_ids, strict=True):
            state = task_states[task_id]
            if state["idx"] is None:  # First time seeing this task_id
                state["idx"] = idx_counter
                state["task"] = task
                idx_counter += 1
            rollout_idx = state["total_rollouts"]
            futures.append(self.process_task_with_retry(task, task_id, rollout_idx, **kwargs))
            state["total_rollouts"] += 1

        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                task_id, rollout_idx, episode = await future

                state = task_states[task_id]
                state["episodes"].append(episode)
                state["completed"] += 1
                pbar.update(1)

        results = []
        sorted_tasks = sorted(task_states.keys(), key=lambda task_id: task_states[task_id]["idx"])
        for task_id in sorted_tasks:
            results.extend(task_states[task_id]["episodes"])

        # Log episodes if logger is provided
        if self.episode_logger is not None:
            try:
                logger.info(f"Logging {len(results)} episodes to step={self.current_step}, mode={self.current_mode}, epoch={self.current_epoch}")
                self.episode_logger.log_episodes_batch(results, self.current_step, self.current_mode, self.current_epoch)
            except Exception as e:
                logger.error(f"Failed to log episodes: {e}")
                import traceback

                traceback.print_exc()

        return results

    async def execute_tasks_verl(self, batch: "DataProto", **kwargs) -> "DataProto":
        """Execute tasks from a Verl DataProto batch and return results.

        Args:
            batch: Verl DataProto containing tasks and metadata.
            **kwargs: Additional arguments passed to execute_tasks.

        Returns:
            DataProto: Transformed results compatible with Verl training.
        """
        await self.rollout_engine.wake_up()

        is_validation = batch.meta_info.get("validate", False)
        if is_validation:
            self.rollout_engine.validate = True
            self.current_mode = "val"
        else:
            self.current_mode = "train"
        tasks = batch.non_tensor_batch["extra_info"].tolist()
        task_ids = batch.non_tensor_batch["task_ids"].tolist()
        results = await self.execute_tasks(tasks, task_ids, **kwargs)  # list of Episodes
        self.rollout_engine.validate = False

        await self.rollout_engine.sleep()

        self.current_mode = "train"
        return self.transform_results_for_verl(results, task_ids)

    def transform_results_for_verl(self, episodes: list[Episode], task_ids: np.ndarray) -> "DataProto":
        """Transform episode results into Verl-compatible DataProto format.

        Args:
            episodes: List of completed episodes from workflow execution.
            task_ids: Array of task identifiers corresponding to episodes.

        Returns:
            DataProto: Formatted data ready for Verl training pipeline.
        """
        # Local import to keep verl optional
        from verl import DataProto
        from verl.utils.torch_functional import pad_sequence_to_length

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
        multi_modal_inputs_list = []
        chat_completions_list = []
        rollout_log_probs_list = []

        for i, episode in enumerate(episodes):
            total_steps = 0

            if episode is None:
                print(f"Episode {i} is None (failed task), dropping it from the batch")
                repeat_counts.append(0)
                continue

            if all(len(trajectory.steps) == 0 for trajectory in episode.trajectories):
                # termination hits before an agent finishes it's first step
                # (e.g., the initial prompt exceeds max_prompt_length or a timeout occurs)
                # we delete the episode from the batch by setting repeat_counts to 0
                print(f"Episode {episode.id} has no valid trajectories, dropping it from the batch")
                repeat_counts.append(0)
                continue

            for trajectory in episode.trajectories:
                name = trajectory.name
                trajectory_id = f"{task_ids[i]}_{name}"  # unique trajectory identifier e.g., 1234567890_solver

                if len(trajectory.steps) == 0:
                    logger.info(f"Trajectory {trajectory_id} has no steps, skipping")
                    continue

                if not self.config.rllm.stepwise_advantage.enable:
                    if len(trajectory.steps) > 1:
                        if not trajectory.is_cumulative():
                            logger.warning(f"Warning: Multi-step trajectory {trajectory_id} is not cumulative, but stepwise mode is not enabled. There could be a token mismatch during trajectory generation.")

                        chat_completions = trajectory.steps[-1].chat_completions
                        chat_completions_list.append(chat_completions)
                        prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask_cumulative(chat_completions)
                        prompts.append(prompt)
                        responses.append(response)
                        traj_mask.append(mask)
                        multi_modal_inputs_list.append({})  # empty dict

                    elif isinstance(trajectory.steps[0].model_output, ModelOutput):
                        step = trajectory.steps[0]
                        # For ModelOutput, use chat_completions if available, otherwise None
                        chat_completions_list.append(step.chat_completions if hasattr(step, "chat_completions") and step.chat_completions else None)

                        prompt_ids = torch.tensor(step.model_output.prompt_ids, dtype=torch.long)
                        prompts.append(prompt_ids)

                        response_ids = torch.tensor(step.model_output.completion_ids, dtype=torch.long)
                        responses.append(response_ids)

                        mask = torch.ones_like(response_ids, dtype=torch.long)
                        traj_mask.append(mask)
                        multi_modal_inputs_list.append(step.model_output.multi_modal_inputs or {})

                        logprobs = torch.tensor(step.model_output.logprobs, dtype=torch.float32)
                        rollout_log_probs_list.append(logprobs)

                    else:
                        chat_completions = trajectory.steps[0].chat_completions
                        chat_completions_list.append(chat_completions)
                        prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask(chat_completions)
                        prompts.append(prompt)
                        responses.append(response)
                        traj_mask.append(mask)
                        multi_modal_inputs_list.append({})  # empty dict

                    step_rewards.append(trajectory.reward)
                    step_ids.append(trajectory_id)
                    n_steps = 1

                else:
                    for step_idx, step in enumerate(trajectory.steps):
                        if isinstance(step.model_output, ModelOutput):
                            # For ModelOutput, use chat_completions if available, otherwise None
                            chat_completions_list.append(step.chat_completions if hasattr(step, "chat_completions") and step.chat_completions else None)
                            prompt_ids = torch.tensor(step.model_output.prompt_ids, dtype=torch.long)
                            prompts.append(prompt_ids)

                            response_ids = torch.tensor(step.model_output.completion_ids, dtype=torch.long)
                            responses.append(response_ids)

                            mask = torch.ones_like(response_ids, dtype=torch.long)
                            traj_mask.append(mask)
                            multi_modal_inputs_list.append(step.model_output.multi_modal_inputs or {})

                            logprobs = torch.tensor(step.model_output.logprobs, dtype=torch.float32)
                            rollout_log_probs_list.append(logprobs)

                        else:
                            chat_completions = step.chat_completions
                            chat_completions_list.append(chat_completions)
                            prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask(chat_completions)
                            prompts.append(prompt)
                            responses.append(response)
                            traj_mask.append(mask)
                            multi_modal_inputs_list.append({})  # empty dict

                        step_rewards.append(step.reward)
                        step_ids.append(f"{trajectory_id}_step{step_idx}")  # unique step identifier e.g., 1234567890_solver_step0

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

        prompt_lengths = torch.as_tensor([len(t) for t in prompts]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in responses]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        if hasattr(self.rollout_engine, "processor") and self.rollout_engine.processor is not None:
            position_ids = self._handle_multimodal_position_ids(
                processor=self.rollout_engine.processor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_modal_inputs=multi_modal_inputs_list,
            )
        else:
            position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        traj_mask = torch.nn.utils.rnn.pad_sequence(traj_mask, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        traj_mask = traj_mask[:, :max_response_length]  # truncate if necessary

        # Place all rewards to last response token of the last_step response
        traj_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        step_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, (traj_reward, step_reward) in enumerate(zip(traj_rewards, step_rewards, strict=False)):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= traj_rewards_batch.shape[1]:
                traj_rewards_batch[i, resp_len - 1] = traj_reward
                step_rewards_batch[i, resp_len - 1] = step_reward

        rollout_log_probs_batch = None
        if rollout_log_probs_list:
            rollout_log_probs_batch = torch.nn.utils.rnn.pad_sequence(
                rollout_log_probs_list,
                batch_first=True,
                padding_value=0.0,
            )
            rollout_log_probs_batch = pad_sequence_to_length(rollout_log_probs_batch, max_response_length, 0.0, left_pad=False)
            rollout_log_probs_batch = rollout_log_probs_batch[:, :max_response_length]

        # compact filtering
        cf = self.config.rllm.compact_filtering
        is_valid = [True] * len(episode_ids)
        if cf.enable:
            for i in range(len(episode_ids)):
                termination_reason = termination_reasons[i]
                if (cf.mask_max_prompt_length_exceeded and termination_reason == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED) or (cf.mask_max_response_length_exceeded and termination_reason == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED) or (cf.mask_env_done and termination_reason == TerminationReason.ENV_DONE) or (cf.mask_max_turns_exceeded and termination_reason == TerminationReason.MAX_TURNS_EXCEEDED) or (cf.mask_timeout and termination_reason == TerminationReason.TIMEOUT) or (cf.mask_unknown and termination_reason == TerminationReason.UNKNOWN) or (cf.mask_error and termination_reason == TerminationReason.ERROR):
                    is_valid[i] = False  # set flag to filter out the episode later (after advantages are computed)

        non_tensors = {
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
            "chat_completions": np.array(chat_completions_list, dtype=object),  # chat completions for distillation
        }

        if any(mm_inputs is not None for mm_inputs in multi_modal_inputs_list):
            non_tensors["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts_batch,
            "responses": response_batch,
            "response_mask": traj_mask,
            "traj_rewards": traj_rewards_batch,
            "step_rewards": step_rewards_batch,
        }

        if rollout_log_probs_batch is not None:
            tensors["rollout_log_probs"] = rollout_log_probs_batch

        return DataProto.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info={
                "repeat_counts": repeat_counts,
            },
        )

    def _handle_multimodal_position_ids(self, processor, input_ids: torch.Tensor, attention_mask: torch.Tensor, multi_modal_inputs: list[dict]) -> torch.Tensor:
        """Handle multimodal position ids calculation. Borrowed from verl.utils.dataset.rl_dataset.py"""
        batch_size = input_ids.shape[0]
        position_ids_list = []

        if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            for i in range(batch_size):
                model_inputs = multi_modal_inputs[i] if i < len(multi_modal_inputs) else {}
                vision_position_ids = get_rope_index(
                    processor,
                    input_ids=input_ids[i],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[i],
                )  # (3, seq_length)
                valid_mask = attention_mask[i].bool()
                text_position_ids = torch.ones((1, len(input_ids[i])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                position_ids_list.append(torch.cat((text_position_ids, vision_position_ids), dim=0))  # (4, seq_length)

        else:
            # Fallback: should not reach here if called correctly
            raise ValueError(f"Unsupported processor type: {processor.__class__.__name__ if processor else None}")

        # Stack all position_ids to form batch: (batch_size, 4, seq_length)
        position_ids = torch.stack(position_ids_list, dim=0)
        return position_ids

    def shutdown(self):
        """Shutdown the workflow engine and cleanup resources."""
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
