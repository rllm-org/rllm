import asyncio
import functools
import inspect
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from tqdm import tqdm

from rllm.agents.agent import Episode, Trajectory
from rllm.engine.rollout import ModelOutput, RolloutEngine
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.misc import colorful_print
from rllm.sdk.data_process import group_steps, trace_to_step
from rllm.sdk.protocol import Trace, TrajectoryView
from rllm.sdk.proxy.proxy_manager import VerlProxyManager
from rllm.sdk.session import SESSION_BACKEND
from rllm.sdk.session.base import wrap_with_session_context
from rllm.sdk.store.sqlite_store import SqliteTraceStore
from rllm.workflows.workflow import TerminationReason

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from verl import DataProto

    from rllm.sdk.tracers import TracerProtocol

logger = logging.getLogger(__name__)


class AgentSdkEngine:
    def __init__(self, agent_run_func: Callable, rollout_engine: RolloutEngine, config=None, n_parallel_tasks: int = 128, retry_limit: int = 3, raise_on_error: bool = True, proxy_config: dict | None = None, tracer: Optional["TracerProtocol"] = None, **kwargs):
        """Initialize SdkEngine for executing agent_run_func on multiple tasks.

        Args:
            agent_run_func: Agent rollout function to execute for each task.
            rollout_engine: Model inference engine (VerlEngine supported).
            config: Training configuration (required for VERL integration).
            n_parallel_tasks: Max parallel workflow instances (default: 128).
            retry_limit: Max retry attempts per failed task (default: 3).
            raise_on_error: Raise on permanent failures (default: True).
            proxy_config: LiteLLM proxy configuration dict:
                - model_name: Model name to expose (required for VERL)
                - proxy_host: Proxy bind address (default: "127.0.0.1")
                - proxy_port: Proxy port (default: 4000)
                - mode: "external" (manual start) or "subprocess" (auto-start)
                - admin_token: Admin API token (default: "my-shared-secret")
                - db_path: SQLite DB path (subprocess mode only)
                - project: Project name for traces (subprocess mode only)
            tracer: Optional tracer for logging.
            **kwargs: Additional arguments.
        """
        self.rollout_engine = rollout_engine
        self.agent_run_func = agent_run_func
        self.config = config  # if training

        self.retry_limit = retry_limit  # number of attempts to retry a task
        self.raise_on_error = raise_on_error
        self.kwargs = kwargs

        self.n_parallel_tasks = n_parallel_tasks
        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel_tasks)
        self.sema = asyncio.Semaphore(self.n_parallel_tasks)

        # Initialize proxy manager for VERL engines
        self.proxy_manager: VerlProxyManager | None = None
        self.rollout_engine_endpoint: str | None = None

        proxy_config = proxy_config or {}

        if isinstance(rollout_engine, VerlEngine):
            self._setup_verl_proxy(proxy_config, tracer)
        else:
            raise NotImplementedError(f"Rollout engine type {type(rollout_engine)} not supported")

        self.wrapped_agent_run_func = wrap_with_session_context(self.agent_run_func, tracer_service_name="agent-sdk-worker")
        self.groupby_key = self.config.rllm.sdk.processing.groupby_key
        self.traj_name_key = self.config.rllm.sdk.processing.traj_name_key
        self.store = SqliteTraceStore(db_path=self.config.rllm.sdk.store.path)

    def _setup_verl_proxy(self, proxy_config: dict, tracer: Optional["TracerProtocol"]) -> None:
        """Setup LiteLLM proxy for VERL rollout engine.

        Initializes VerlProxyManager and starts proxy in subprocess or external mode.
        Proxy handles trace collection from vLLM servers and persists to SQLite.

        When using OpenTelemetry-based sessions, sync storage mode is required to ensure
        synchronization between tracer persistence and session reads.
        """
        model_name = proxy_config.get("model_name")
        if not model_name:
            logger.warning("No model_name provided in proxy_config. Proxy manager will not be initialized. Provide proxy_config={'model_name': 'your-model'} to enable proxy.")
            return

        proxy_host = proxy_config.get("proxy_host", "127.0.0.1")
        proxy_port = proxy_config.get("proxy_port", 4000)
        proxy_mode = proxy_config.get("mode", "external")
        admin_token = proxy_config.get("admin_token", "my-shared-secret")

        # Check if using OpenTelemetry session backend - requires sync storage mode
        requires_sync_storage = SESSION_BACKEND == "opentelemetry"

        if requires_sync_storage and proxy_mode == "external":
            logger.warning("OpenTelemetry-based sessions require synchronous storage mode for proper synchronization. When using external proxy mode, ensure the proxy is started with --sync-tracer flag. Alternatively, use proxy_mode='subprocess' to automatically enable sync storage. Without sync storage, there may be synchronization issues between tracer persistence and session reads.")

        add_logprobs = proxy_config.get("add_logprobs", False)

        self.proxy_manager = VerlProxyManager(
            rollout_engine=self.rollout_engine,
            model_name=model_name,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            admin_token=admin_token,
            proxy_access_log=False,
            add_logprobs=add_logprobs,
        )

        self.rollout_engine_endpoint = self.proxy_manager.get_proxy_url()

        print(f"Initialized VerlProxyManager with {len(self.proxy_manager.get_server_addresses())} vLLM replicas. Proxy endpoint: {self.rollout_engine_endpoint}")

        # Build config once per setup so both modes stay in sync
        config_payload = self.proxy_manager.build_proxy_config()

        # Start proxy based on mode
        if proxy_mode == "subprocess":
            # Start subprocess, wait for server, then reload config
            db_path = proxy_config.get("db_path")
            project = proxy_config.get("project", "rllm-agent-sdk")
            # Enable sync storage when using OpenTelemetry sessions
            sync_tracer = requires_sync_storage
            if sync_tracer:
                logger.info("Enabling synchronous tracer persistence for OpenTelemetry session backend")
            self.proxy_manager.start_proxy_subprocess(
                config=config_payload,
                db_path=db_path,
                project=project,
                sync_tracer=sync_tracer,
                add_logprobs=add_logprobs,
                add_return_token_ids=True,
            )
        elif proxy_mode == "external":
            # Reload external proxy with the generated configuration
            self.proxy_manager.reload_proxy_config(config=config_payload)
        else:
            raise ValueError(f"Unknown proxy mode: {proxy_mode}. Must be 'external' or 'subprocess'")

    async def initialize_pool(self):
        """Initialize semaphore for controlling concurrent task execution.

        Creates asyncio semaphore to limit parallel execution.
        Idempotent - safe to call multiple times (returns early if already initialized).
        """
        # if self.sema is not None:
        #     return
        # Use a semaphore instead of queue to control concurrency without deadlock
        # self.sema = asyncio.Semaphore(self.n_parallel_tasks)
        pass

    async def _execute_with_exception_handling(self, func, task, task_id, rollout_idx, attempt_idx, **kwargs):
        # Format: "{task_id}:{rollout_idx}:{attempt_idx}"
        # This uniquely identifies each rollout attempt
        metadata = {"session_name": f"{task_id}:{rollout_idx}:{attempt_idx}", "task": task}
        try:
            if inspect.iscoroutinefunction(self.wrapped_agent_run_func):
                output, session_uid = await func(metadata, **task, **kwargs)
                return True, output, session_uid
            else:
                loop = asyncio.get_event_loop()
                bound_func = functools.partial(func, metadata, **task, **kwargs)
                output, session_uid = await loop.run_in_executor(self.executor, bound_func)
                return True, output, session_uid
        except Exception:
            import traceback

            error_tb = traceback.format_exc()
            logger.error(f"[{task_id}:{rollout_idx}:{attempt_idx}] Rollout failed: {error_tb}")
            return False, error_tb, None

    async def process_task_with_retry(self, task: dict, task_id: str, rollout_idx: int, **kwargs) -> tuple[str, int, int, float, str]:
        """Process single task rollout with automatic retry on failure.

        Executes task with retry logic, using wrapped agent function.
        Semaphore controls concurrency to prevent resource exhaustion.
        Session name format: "{task_id}:{rollout_idx}:{retry_attempt}".

        Args:
            task: Task specification dict.
            task_id: Unique task identifier.
            rollout_idx: Rollout index for this task.
            **kwargs: Additional args passed to agent function.

        Returns:
            Tuple of (task_id, rollout_idx, retry_attempt, reward/output, session_uid).

        Raises:
            Exception: If task fails permanently after retry_limit attempts.
        """
        # Use semaphore to control concurrent executions and prevent deadlock
        async with self.sema:  # agent_queue is now a Semaphore
            func = self.wrapped_agent_run_func
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}:{retry_attempt}"
                success, output, session_uid = await self._execute_with_exception_handling(func, task=task, task_id=task_id, rollout_idx=rollout_idx, attempt_idx=retry_attempt, **kwargs)
                if success and isinstance(output, float | int | bool):
                    colorful_print(f"[{uid}] Rollout completed with reward: {float(output)}", fg="green" if float(output) > 0 else "yellow")
                    return task_id, rollout_idx, retry_attempt, float(output), session_uid
                elif success and isinstance(output, list):
                    assert all(isinstance(t, TrajectoryView) for t in output), "Must be a list of TrajectoryView"
                    return task_id, rollout_idx, retry_attempt, output, session_uid
                if retry_attempt < self.retry_limit:
                    print(f"[{uid}] Rollout failed on attempt {retry_attempt}/{self.retry_limit}, retrying...")
                    continue
                raise Exception(f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts: {output}")

    async def _execute_tasks(self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs) -> list[Episode]:
        """Execute multiple tasks asynchronously with retry logic and trace collection.

        Launches all tasks concurrently, collects traces from SQLite after completion,
        groups traces into trajectories, and builds Episode objects with rewards.

        Args:
            tasks: List of task specification dicts.
            task_ids: Optional task IDs (UUIDs generated if None).
            **kwargs: Additional args passed to task processing.

        Returns:
            List of Episode objects with trajectories and rewards.
        """
        if self.sema is None:
            await self.initialize_pool()

        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        task_states = defaultdict(lambda: {"idx": None, "task": None, "episodes": [], "completed": 0, "total_rollouts": 0, "is_complete": False})

        # Capture rollout start time BEFORE launching tasks to ensure all traces are included
        rollout_start_time = time.time()

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

        rollout_session_names = set()
        session_uids = set()
        outputs = dict()
        start_time = time.time()
        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                task_id, rollout_idx, retry_attempt, output, session_uid = await future
                session_uids.add(session_uid)
                rollout_session_names.add(f"{task_id}:{rollout_idx}:{retry_attempt}")
                outputs[f"{task_id}:{rollout_idx}:{retry_attempt}"] = output
                state = task_states[task_id]
                state["completed"] += 1
                pbar.update(1)

        print(f"Total time for generating trajectories: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        flush_success = await self.flush_traces(timeout=60.0)
        flush_time = time.time() - start_time

        all_traces = []
        collect_trajectory_start = time.time()
        for session_uid in session_uids:
            traces = await self.store.get_by_session_uid(session_uid, since=rollout_start_time)
            all_traces.extend(traces)
        collect_sqlite_time = time.time() - collect_trajectory_start

        traces_by_session_name = {}
        for session_name in rollout_session_names:
            traces_by_session_name[session_name] = []

        for trace in all_traces:
            session_name = trace.data.get("session_name", None)
            if not session_name or session_name not in rollout_session_names:
                continue
            trace_obj = Trace(**trace.data)
            traces_by_session_name[session_name].append((trace.id, trace_obj))

        for session_name, traces in traces_by_session_name.items():
            steps = [trace_to_step(entry[1]) for entry in traces]
            step_id_to_step = {entry[0]: step for entry, step in zip(traces, steps, strict=False)}

            task_id = session_name.split(":")[0]
            retry_attempt = int(session_name.split(":")[2])

            output = outputs[session_name]
            if isinstance(output, float):
                trajectories = group_steps(steps, by=self.groupby_key, name_key=self.traj_name_key)
                # fill reward for each trajectory using the final reward
                for trajectory in trajectories:
                    trajectory.reward = output
                is_correct = output >= 1.0
            else:
                # assemble and assign rewards based on user provide traj_proto
                trajectories = []
                for traj_proto in output:
                    steps_no_rw = [step_id_to_step.get(step.id, None) for step in traj_proto.steps]
                    for step_proto, step in zip(traj_proto.steps, steps_no_rw, strict=True):
                        if step is None:
                            print(f"Step {step_proto.id} not found in step_id_to_step, persistant storage failed?")
                            continue
                        step.reward = step_proto.reward
                    trajectories.append(
                        Trajectory(
                            name=traj_proto.name,
                            steps=steps_no_rw,
                            reward=traj_proto.reward,
                        )
                    )
                is_correct = trajectories[-1].reward >= 1.0 if len(trajectories) > 0 else False

            steps_used = sum(len(trajectory.steps) for trajectory in trajectories)

            # Collect all response and prompt lengths for metrics
            all_response_lens = [len(step.model_output.completion_ids) for trajectory in trajectories for step in trajectory.steps]
            all_prompt_lens = [len(step.model_output.prompt_ids) for trajectory in trajectories for step in trajectory.steps]

            metrics = {
                "retry_attempt": retry_attempt,
                "empty": int(len(steps) == 0),
                "flush_success": int(flush_success),
                "num_trajectories": len(trajectories),
                "steps_collected": len(steps),
                "steps_used": steps_used,
                "mean_response_len": sum(all_response_lens) / len(all_response_lens) if all_response_lens else 0,
                "max_response_len": max(all_response_lens, default=0),
                "min_response_len": min(all_response_lens, default=0),
                "max_prompt_len": max(all_prompt_lens, default=0),
                "min_prompt_len": min(all_prompt_lens, default=0),
                "collect_sqlite_time": collect_sqlite_time,
                "flush_time": flush_time,
            }

            episode = Episode(id=session_name, is_correct=is_correct, trajectories=trajectories, metrics=metrics)
            task_states[task_id]["episodes"].append(episode)

        results = []
        sorted_tasks = sorted(task_states.keys(), key=lambda task_id: task_states[task_id]["idx"])
        for task_id in sorted_tasks:
            results.extend(task_states[task_id]["episodes"])
        return results

    async def execute_tasks(self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs) -> list[Episode]:
        for _ in range(3):
            results = [None] * len(tasks)
            try:
                results = await self._execute_tasks(tasks, task_ids, **kwargs)
            except Exception as e:
                print(f"Error in execute_tasks: {e}, retrying...")

            error_count = 0
            for episode in results:
                if episode is None:
                    error_count += 1
                    continue
                if episode.trajectories is None or len(episode.trajectories) == 0:
                    error_count += 1
            if error_count / len(results) > 0.01:
                print(f"Too many errors in execute_tasks: {error_count} / {len(results)} > 0.01, sleeping for 120s before retrying...")
                await asyncio.sleep(120.0)
            else:
                return results
        raise Exception("Failed to execute tasks after 3 retries")

    async def flush_traces(self, timeout: float = 30.0) -> bool:
        """Flush all traces to storage via LiteLLM proxy.

        Sends flush signal to proxy, which persists all queued traces to SQLite.
        Ensures traces are available for retrieval before returning.

        Args:
            timeout: Max wait time for flush operation (default: 30.0 seconds).

        Returns:
            True if flush succeeds, False otherwise.
        """
        if not self.proxy_manager:
            logger.warning("No proxy manager available, cannot flush traces")
            return False

        logger.info("Flushing traces via proxy manager (timeout=%s)", timeout)
        success = await self.proxy_manager.flush_tracer(timeout=timeout)

        if success:
            logger.info("Successfully flushed all traces")
        else:
            logger.warning("Failed to flush traces")

        return success

    async def execute_tasks_verl(self, batch: "DataProto", **kwargs) -> "DataProto":
        """Execute tasks from VERL batch and transform results for training.

        Extracts tasks from DataProto, executes workflows, collects episodes,
        and transforms to VERL-compatible format with tokenized prompts/responses.

        Args:
            batch: VERL DataProto containing tasks in non_tensor_batch["extra_info"].
            **kwargs: Additional args passed to execute_tasks.

        Returns:
            DataProto with training-ready tensors (prompts, responses, rewards, masks).
        """
        # TODO: later probably should make the `wake_up` and `sleep` methods in base class to be async
        if isinstance(self.rollout_engine, VerlEngine):
            await self.rollout_engine.wake_up()
        else:
            self.rollout_engine.wake_up()

        if batch.meta_info.get("validate", False):
            self.rollout_engine.validate = True
        tasks = batch.non_tensor_batch["extra_info"].tolist()
        task_ids = batch.non_tensor_batch["task_ids"].tolist()
        episodes = await self.execute_tasks(tasks, task_ids, **kwargs)  # list of Episodes
        self.rollout_engine.validate = False

        if isinstance(self.rollout_engine, VerlEngine):
            await self.rollout_engine.sleep()
        else:
            self.rollout_engine.sleep()
        return self.transform_results_for_verl(episodes, task_ids)

    def transform_results_for_verl(self, episodes: list[Episode], task_ids: np.ndarray) -> "DataProto":
        """Transform episodes into VERL-compatible DataProto format.

        Args:
            episodes: List of Episodes from workflow execution.
            task_ids: Array of task IDs corresponding to episodes.

        Returns:
            DataProto with tensors (input_ids, attention_mask, responses, rewards, etc.)
            and metadata (episode_ids, trajectory_ids, step_ids, termination_reasons).
        """
        # Local import to keep verl optional
        from verl import DataProto
        from verl.utils.torch_functional import pad_sequence_to_length

        prompts = []
        responses = []
        rollout_logprobs = []
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
                trajectory_id = f"{task_ids[i]}_{name}"  # e.g., "abc123_solver", "abc123_judge"

                if len(trajectory.steps) == 0:
                    print(f"Trajectory {trajectory_id} has no steps, skipping")
                    continue

                # if not self.config.rllm.stepwise_advantage.enable:
                #     if len(trajectory.steps) > 1:
                #         if not trajectory.is_cumulative():
                #             logger.warning(f"Warning: Multi-step trajectory {trajectory_id} is not cumulative, but stepwise mode is not enabled. There could be a token mismatch during trajectory generation.")

                #         chat_completions = trajectory.steps[-1].chat_completions
                #         prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask_cumulative(chat_completions)
                #         prompts.append(prompt)
                #         responses.append(response)
                #         traj_mask.append(mask)

                #     elif isinstance(trajectory.steps[0].model_output, ModelOutput):
                #         step = trajectory.steps[0]

                #         prompt_ids = torch.tensor(step.model_output.prompt_ids, dtype=torch.long)
                #         prompts.append(prompt_ids)

                #         response_ids = torch.tensor(step.model_output.completion_ids, dtype=torch.long)
                #         responses.append(response_ids)

                #         mask = torch.ones_like(response_ids, dtype=torch.long)
                #         traj_mask.append(mask)

                #     else:
                #         chat_completions = trajectory.steps[0].chat_completions
                #         prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask(chat_completions)
                #         prompts.append(prompt)
                #         responses.append(response)
                #         traj_mask.append(mask)

                #     step_rewards.append(trajectory.reward)
                #     step_ids.append(trajectory_id)
                #     n_steps = 1

                # else:
                # TODO: auto merge the steps if they share some prefix
                max_prompt_length = self.config.data.max_prompt_length
                valid_step_count = 0
                for step_idx, step in enumerate(trajectory.steps):
                    if isinstance(step.model_output, ModelOutput):
                        prompt_ids = torch.tensor(step.model_output.prompt_ids, dtype=torch.long)

                        # Skip steps with overlong prompts to avoid OOD (train/inference mismatch)
                        # During rollout, model saw full prompt; truncating would create OOD data
                        if len(prompt_ids) > max_prompt_length:
                            logger.warning(f"Skipping step {step_idx} of trajectory {trajectory_id}: prompt length {len(prompt_ids)} > max_prompt_length {max_prompt_length}")
                            continue

                        prompts.append(prompt_ids)

                        response_ids = torch.tensor(step.model_output.completion_ids, dtype=torch.long)
                        responses.append(response_ids)

                        rollout_logprob = torch.tensor(step.model_output.logprobs, dtype=torch.float32)
                        rollout_logprobs.append(rollout_logprob)

                        mask = torch.ones_like(response_ids, dtype=torch.long)
                        traj_mask.append(mask)

                        # else:
                        #     chat_completions = step.chat_completions
                        #     prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask(chat_completions)
                        #     prompts.append(prompt)
                        #     responses.append(response)
                        #     traj_mask.append(mask)

                        step_rewards.append(step.reward)
                        step_ids.append(f"{trajectory_id}_step{step_idx}")  # e.g., "abc123_solver_step0", "abc123_judge_step1"

                        valid_step_count += 1

                # Use valid_step_count (steps not filtered due to overlong prompts)
                n_steps = valid_step_count
                if n_steps == 0:
                    logger.warning(f"Trajectory {trajectory_id} has no valid steps after filtering overlong prompts, skipping")
                    continue

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
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        traj_mask = torch.nn.utils.rnn.pad_sequence(traj_mask, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        traj_mask = traj_mask[:, :max_response_length]  # truncate if necessary

        # Pad rollout_logprobs to match response_batch shape
        rollout_logprobs_batch = None
        if rollout_logprobs and all(len(logprobs) == len(response) for logprobs, response in zip(rollout_logprobs, responses, strict=False)):
            rollout_logprobs_batch = torch.nn.utils.rnn.pad_sequence(
                rollout_logprobs,
                batch_first=True,
                padding_value=-100.0,
            )
            rollout_logprobs_batch = pad_sequence_to_length(rollout_logprobs_batch, max_response_length, -100.0, left_pad=False)
            rollout_logprobs_batch = rollout_logprobs_batch[:, :max_response_length]  # truncate if necessary
            rollout_logprobs_batch = rollout_logprobs_batch.to(torch.float32)

        # Place all rewards to last response token of the last_step response
        traj_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        step_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, (traj_reward, step_reward) in enumerate(zip(traj_rewards, step_rewards, strict=False)):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= traj_rewards_batch.shape[1]:
                traj_rewards_batch[i, resp_len - 1] = traj_reward
                step_rewards_batch[i, resp_len - 1] = step_reward

        # compact filtering
        cf = self.config.rllm.compact_filtering
        is_valid = [True] * len(episode_ids)
        if cf.enable:
            for i in range(len(episode_ids)):
                termination_reason = termination_reasons[i]
                if (cf.mask_max_prompt_length_exceeded and termination_reason == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED) or (cf.mask_max_response_length_exceeded and termination_reason == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED) or (cf.mask_env_done and termination_reason == TerminationReason.ENV_DONE) or (cf.mask_max_turns_exceeded and termination_reason == TerminationReason.MAX_TURNS_EXCEEDED) or (cf.mask_timeout and termination_reason == TerminationReason.TIMEOUT) or (cf.mask_unknown and termination_reason == TerminationReason.UNKNOWN) or (cf.mask_error and termination_reason == TerminationReason.ERROR):
                    is_valid[i] = False  # set flag to filter out the episode later (after advantages are computed)

        # Build tensors dict, conditionally include rollout_log_probs if available
        tensors_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts_batch,
            "responses": response_batch,
            "response_mask": traj_mask,
            "traj_rewards": traj_rewards_batch,
            "step_rewards": step_rewards_batch,
        }
        if rollout_logprobs_batch is not None:
            tensors_dict["rollout_log_probs"] = rollout_logprobs_batch

        return DataProto.from_dict(
            tensors=tensors_dict,
            non_tensors={
                # episode_ids: Format "task_id:rollout_idx:retry_attempt" (e.g., "abc123:0:1")
                "episode_ids": np.array(episode_ids),
                # trajectory_ids: Format "task_id_trajectory_name" (e.g., "abc123_solver")
                # Multiple trajectories can have the same trajectory_id
                "trajectory_ids": np.array(trajectory_ids),
                # step_ids: Format "task_id_trajectory_name_step{idx}" (e.g., "abc123_solver_step0")
                "step_ids": np.array(step_ids),
                # batch_ids: Unique identifier for each training batch
                "batch_ids": np.array([str(uuid.uuid4())] * len(episode_ids)),
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
        """Shutdown the workflow engine and cleanup resources."""
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

        # Shutdown proxy subprocess if running
        if hasattr(self, "proxy_manager") and self.proxy_manager is not None:
            self.proxy_manager.shutdown_proxy()
