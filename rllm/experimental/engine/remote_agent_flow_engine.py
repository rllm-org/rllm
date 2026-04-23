"""RemoteAgentFlowEngine: submits tasks to remote runtimes, merges rewards + gateway traces.

Backend-agnostic engine that works with any RemoteAgentRuntime implementation.
Reuses GatewayManager for session/trace management and trace_record_to_step for
converting gateway traces to training Steps.
"""

import asyncio
import logging
import uuid
from collections import defaultdict

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.experimental.engine.gateway_manager import GatewayManager
from rllm.experimental.engine.remote_runtime.protocol import (
    RemoteAgentRuntime,
    RemoteTaskResult,
    TaskSubmission,
)
from rllm.experimental.engine.trace_converter import compute_step_metrics, trace_record_to_step
from rllm.utils.episode_logger import EpisodeLogger
from rllm.workflows.workflow import TerminationReason

logger = logging.getLogger(__name__)


class RemoteAgentFlowEngine:
    """Submits tasks to remote runtimes and builds Episodes from gateway traces + rewards."""

    def __init__(
        self,
        runtime: RemoteAgentRuntime,
        gateway: GatewayManager,
        session_timeout: float = 900.0,
        n_parallel_tasks: int = 128,
        episode_logger: EpisodeLogger | None = None,
    ) -> None:
        self.runtime = runtime
        self.gateway = gateway
        self.session_timeout = session_timeout
        self.n_parallel_tasks = n_parallel_tasks
        self.episode_logger = episode_logger
        self._semaphore = asyncio.Semaphore(n_parallel_tasks)

        # Training step tracking (set by set_training_step)
        self.current_step = 0
        self.current_epoch = 0
        self.current_mode = "train"

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0) -> None:
        self.current_step = step
        self.current_mode = mode
        self.current_epoch = epoch

    async def execute_tasks(
        self,
        tasks: list[dict],
        task_ids: list[str] | None = None,
        is_validation: bool = False,
        **kwargs,
    ) -> list[Episode]:
        """Submit tasks to remote runtime, gather results, build Episodes from gateway traces."""
        from tqdm.asyncio import tqdm

        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        task_id_counter: dict[str, int] = defaultdict(int)
        results: list[Episode | None] = [None] * len(tasks)

        futures = []
        for idx, (task, task_id) in enumerate(zip(tasks, task_ids, strict=True)):
            rollout_idx = task_id_counter[task_id]
            task_id_counter[task_id] += 1
            futures.append(self.process_task_with_retry(task, task_id, rollout_idx, idx, is_validation=is_validation))

        for future in tqdm(asyncio.as_completed(futures), total=len(futures), desc="rollouts"):
            task_id, rollout_idx, idx, episode = await future
            results[idx] = episode

        episodes: list[Episode] = results  # type: ignore[assignment]

        # Log episodes if logger is provided
        if self.episode_logger is not None:
            try:
                self.episode_logger.log_episodes_batch(
                    episodes,
                    self.current_step,
                    self.current_mode,
                    self.current_epoch,
                )
            except Exception as e:
                logger.error("Failed to log episodes: %s", e)

        return episodes

    async def process_task_with_retry(
        self,
        task: dict,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        **kwargs,
    ) -> tuple[str, int, int, Episode]:
        """Process a single task with concurrency control."""
        async with self._semaphore:
            uid = f"{task_id}:{rollout_idx}"
            session_id = str(uuid.uuid4())
            is_validation = kwargs.get("is_validation", False)

            await self.gateway.acreate_session(session_id, is_validation=is_validation)
            session_url = self.gateway.get_session_url(session_id)

            submission = TaskSubmission(
                task=task,
                session_id=session_id,
                task_id=task_id,
                inference_url=session_url,
            )
            results = await self.runtime.execute_tasks([submission], timeout=self.session_timeout)
            result = results[0]

            if not result.finished:
                logger.warning("[%s] Remote task failed (assigning reward=0): %s", uid, result.error)
                result.reward = 0.0

            traces = await self.gateway.aget_traces(session_id)
            episode = _build_episode(traces, result, uid, task)
            if result.metadata:
                episode.metadata.update(result.metadata)
            if not result.finished:
                episode.metadata["error"] = {"message": result.error or "Unknown error"}

            # Delete traces from gateway DB to prevent unbounded growth
            await self.gateway.adelete_session(session_id)

            return task_id, rollout_idx, result_idx, episode

    def shutdown(self) -> None:
        """No local resources to clean up (runtime shutdown is separate)."""
        pass


def _build_episode(
    traces: list,
    result: RemoteTaskResult,
    uid: str,
    task: dict,
) -> Episode:
    """Build an Episode from gateway traces and remote reward.

    Converts all traces to training Steps via trace_record_to_step(),
    creates a single Trajectory with the remote reward, and computes metrics.
    """
    # Convert traces to training steps
    training_steps: list[Step] = []
    if traces:
        training_steps = [trace_record_to_step(t) for t in traces]

    # Create trajectory with all steps and remote reward
    trajectories = []
    if training_steps:
        trajectories.append(
            Trajectory(
                name="default",
                task=task,
                steps=training_steps,
                reward=result.reward,
            )
        )
    elif result.reward is not None:
        # No traces but we have a reward — create empty trajectory
        trajectories.append(
            Trajectory(
                name="default",
                task=task,
                steps=[],
                reward=result.reward,
            )
        )

    # Compute metrics
    metrics = compute_step_metrics(trajectories)
    metrics["empty"] = int(len(traces) == 0)
    metrics["steps_collected"] = len(traces)

    is_correct = bool(result.reward and result.reward > 0)

    return Episode(
        id=uid,
        task=task,
        is_correct=is_correct,
        trajectories=trajectories,
        metrics=metrics,
        termination_reason=result.termination_reason or TerminationReason.UNKNOWN,
    )
