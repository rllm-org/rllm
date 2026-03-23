"""AgentFlowEngine: runs AgentFlows with gateway-mediated trace capture.

Replaces UnifiedWorkflowEngine for the gateway code path. Executes AgentFlow
instances in parallel, retrieves TraceRecords from the gateway, and enriches
the lightweight Episodes with token-level data for training.
"""

from __future__ import annotations

import asyncio
import logging
import resource
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from tqdm import tqdm

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.experimental.engine.trace_converter import trace_record_to_step
from rllm.experimental.eval.types import AgentConfig, EvalOutput, Task, run_agent_flow
from rllm.utils import colorful_print
from rllm.workflows.workflow import TerminationReason

if TYPE_CHECKING:
    from rllm_model_gateway.models import TraceRecord

    from rllm.experimental.engine.gateway_manager import GatewayManager
    from rllm.experimental.eval.types import AgentFlow, Evaluator
    from rllm.utils.episode_logger import EpisodeLogger

logger = logging.getLogger(__name__)

_MIN_FD_LIMIT = 8192


def _raise_fd_limit(target: int = _MIN_FD_LIMIT) -> None:
    """Best-effort raise of the process soft file-descriptor limit.

    Training with many parallel agent flows (each opening HTTP connections
    through the gateway) can easily exceed the default 1024 FD soft limit.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < target:
            new_soft = min(target, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            logger.info("Raised NOFILE soft limit from %d to %d (hard=%d)", soft, new_soft, hard)
    except (ValueError, OSError) as e:
        logger.warning("Could not raise file descriptor limit: %s", e)


class AgentFlowEngine:
    """Executes AgentFlows with gateway-mediated trace capture."""

    def __init__(
        self,
        agent_flow: AgentFlow,
        evaluator: Evaluator,
        gateway: GatewayManager,
        model: str,
        n_parallel_tasks: int = 128,
        retry_limit: int = 3,
        raise_on_error: bool = True,
        episode_logger: EpisodeLogger | None = None,
    ) -> None:
        self.agent_flow = agent_flow
        self.evaluator = evaluator
        self.gateway = gateway
        self.model = model
        self.n_parallel_tasks = n_parallel_tasks
        self.retry_limit = retry_limit
        self.raise_on_error = raise_on_error
        self.episode_logger = episode_logger
        self.executor = ThreadPoolExecutor(max_workers=n_parallel_tasks)

        # Raise the file descriptor limit to avoid "Too many open files" when
        # running many parallel agent flows with individual HTTP clients.
        _raise_fd_limit()

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
        """Run AgentFlows on tasks, capture traces, enrich Episodes.

        For each task:
        1. Create gateway session (uid = "{task_id}:{rollout_idx}")
        2. Build AgentConfig with base_url = gateway session URL
        3. Run agent_flow.run(Task(data=task), config) -> lightweight Episode
        4. Evaluate with evaluator
        5. Retrieve TraceRecords from gateway
        6. Enrich Episode: match traces to trajectories, add token data
        7. Return training-ready Episode
        """
        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        task_id_counter: dict[str, int] = defaultdict(int)
        results: list[Episode | None] = [None] * len(tasks)

        futures = []
        for idx, (task, task_id) in enumerate(zip(tasks, task_ids, strict=True)):
            rollout_idx = task_id_counter[task_id]
            task_id_counter[task_id] += 1
            futures.append(self._process_task_with_retry(task, task_id, rollout_idx, idx, is_validation=is_validation))

        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                task_id, rollout_idx, idx, episode = await future
                results[idx] = episode
                pbar.update(1)

        ordered_results: list[Episode] = results  # type: ignore[assignment]

        # Log episodes if logger is provided
        if self.episode_logger is not None:
            try:
                self.episode_logger.log_episodes_batch(
                    ordered_results,
                    self.current_step,
                    self.current_mode,
                    self.current_epoch,
                )
            except Exception as e:
                logger.error("Failed to log episodes: %s", e)

        return ordered_results

    async def _process_task_with_retry(
        self,
        task: dict,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        is_validation: bool = False,
    ) -> tuple[str, int, int, Episode]:
        """Process a single task with retry logic."""
        for retry_attempt in range(1, self.retry_limit + 1):
            uid = f"{task_id}:{rollout_idx}"
            try:
                episode = await self._run_single(task, uid, is_validation=is_validation)
                episode.id = uid
                episode.task = task

                # Display rewards
                reward_strs = []
                for traj in episode.trajectories:
                    reward = "N/A"
                    if traj.reward is not None:
                        reward = f"{traj.reward:.1f}"
                    elif len(traj.steps) > 0:
                        reward = f"{traj.steps[-1].reward:.1f}"
                    reward_strs.append(f"{traj.name}: {reward}")
                colorful_print(
                    f"[{uid}] Rollout completed. Rewards: [{', '.join(reward_strs)}], Termination: {episode.termination_reason}",
                    fg="green" if episode.is_correct else "yellow",
                )

                return task_id, rollout_idx, result_idx, episode

            except Exception as e:
                logger.error("[%s] Attempt %d/%d failed: %s", uid, retry_attempt, self.retry_limit, e)
                if retry_attempt < self.retry_limit:
                    continue

                if self.raise_on_error:
                    raise

                # Return an error episode
                return (
                    task_id,
                    rollout_idx,
                    result_idx,
                    Episode(
                        id=uid,
                        task=task,
                        is_correct=False,
                        termination_reason=TerminationReason.ERROR,
                        metadata={"error": {"message": str(e)}},
                    ),
                )

        # Should not reach here, but satisfy type checker
        raise RuntimeError(f"[{uid}] Exhausted all retries")

    async def _run_single(self, task: dict, uid: str, is_validation: bool = False) -> Episode:
        """Run a single AgentFlow task: execute, evaluate, enrich."""
        loop = asyncio.get_event_loop()

        # 1. Create gateway session (run in executor to avoid blocking event loop)
        await loop.run_in_executor(
            self.executor,
            self.gateway.create_session,
            uid,
        )
        session_url = self.gateway.get_session_url(uid)

        # 2. Build config
        config = AgentConfig(
            base_url=session_url,
            model=self.model,
            session_uid=uid,
        )

        # 3. Run agent flow (prefers arun if available, else run in executor)
        logger.debug("[%s] Starting agent flow at %s", uid, session_url)
        task_obj = Task(data=task)
        episode = await run_agent_flow(self.agent_flow, task_obj, config, executor=self.executor)
        logger.debug("[%s] Agent flow completed, %d trajectories", uid, len(episode.trajectories))

        # 4. Evaluate
        eval_output: EvalOutput = await loop.run_in_executor(
            self.executor,
            self.evaluator.evaluate,
            task,
            episode,
        )

        # Apply reward to trajectories that don't already have one.
        # Evaluators for multi-trajectory flows (e.g. solver-judge) may set
        # per-trajectory rewards directly on the episode; those are preserved.
        for traj in episode.trajectories:
            if traj.reward is None:
                traj.reward = eval_output.reward
        episode.is_correct = eval_output.is_correct

        # 5. Retrieve traces from gateway (run in executor to avoid blocking event loop)
        traces = await loop.run_in_executor(
            self.executor,
            self.gateway.get_traces,
            uid,
        )

        # 6. Enrich episode with token data
        enriched = self._enrich_episode(episode, traces, uid, task)

        # Attach eval metrics
        enriched.metrics.update(eval_output.metadata)
        for signal in eval_output.signals:
            enriched.metrics[signal.name] = signal.value

        enriched.termination_reason = TerminationReason.ENV_DONE
        return enriched

    def _enrich_episode(
        self,
        episode: Episode,
        traces: list[TraceRecord],
        uid: str,
        task: dict,
    ) -> Episode:
        """Merge gateway traces into agent's lightweight Episode.

        Matching strategy (Option A — positional):
        - Traces are ordered chronologically
        - Walk through trajectories in order, match each step to the next
          trace by position
        - Create training Steps from traces, preserve rewards/done flags
          from agent Steps
        """
        if not traces:
            logger.warning("[%s] No traces found — returning episode without token data", uid)
            return episode

        # Convert all traces to training steps
        training_steps = [trace_record_to_step(t) for t in traces]

        # Build enriched trajectories
        enriched_trajectories: list[Trajectory] = []
        trace_idx = 0

        for traj in episode.trajectories:
            traj_steps: list[Step] = []

            if traj.steps:
                # Match agent steps to traces positionally
                for agent_step in traj.steps:
                    if trace_idx < len(training_steps):
                        step = training_steps[trace_idx]
                        # Preserve reward and done from agent's step
                        step.reward = agent_step.reward
                        step.done = agent_step.done
                        trace_idx += 1
                    else:
                        # No more traces — keep original step
                        step = agent_step
                    traj_steps.append(step)
            else:
                # No agent steps — assign all remaining traces to this trajectory
                # (common for single-trajectory agents that don't populate steps)
                remaining = training_steps[trace_idx:]
                trace_idx += len(remaining)
                traj_steps = remaining

            enriched_trajectories.append(
                Trajectory(
                    uid=traj.uid,
                    name=traj.name,
                    task=traj.task or task,
                    steps=traj_steps,
                    reward=traj.reward,
                    metadata=traj.metadata,
                )
            )

        # If there are unmatched traces and no trajectories existed, create one
        if not episode.trajectories and traces:
            enriched_trajectories = [
                Trajectory(
                    name="default",
                    task=task,
                    steps=training_steps,
                )
            ]

        # Compute metrics
        all_response_lens = [len(s.response_ids) for t in enriched_trajectories for s in t.steps]
        all_prompt_lens = [len(s.prompt_ids) for t in enriched_trajectories for s in t.steps]
        metrics = {
            "empty": int(len(traces) == 0),
            "num_trajectories": len(enriched_trajectories),
            "steps_collected": len(traces),
            "steps_used": sum(len(t.steps) for t in enriched_trajectories),
            "mean_response_len": (sum(all_response_lens) / len(all_response_lens) if all_response_lens else 0),
            "max_response_len": max(all_response_lens, default=0),
            "min_response_len": min(all_response_lens, default=0),
            "max_prompt_len": max(all_prompt_lens, default=0),
            "min_prompt_len": min(all_prompt_lens, default=0),
        }
        metrics.update(episode.metrics)

        return Episode(
            id=uid,
            task=task,
            is_correct=episode.is_correct,
            trajectories=enriched_trajectories,
            metrics=metrics,
            metadata=episode.metadata,
            termination_reason=episode.termination_reason,
            artifacts=episode.artifacts,
        )

    def shutdown(self) -> None:
        """Shutdown the engine and cleanup resources."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
