"""AgentFlowEngine: runs AgentFlows with gateway-mediated trace capture.

Single execution engine for both training and eval. Each rollout:

1. ``hooks.setup(task, agent_flow, uid)`` runs (if hooks provided) — eval
   uses this to create a per-task sandbox + resolve a per-task verifier;
   training leaves hooks unset.
2. A gateway session is created.
3. The agent flow runs against the gateway session URL.
4. Traces are fetched and the Episode is enriched with token-level Steps.
5. The evaluator scores the enriched Episode (per-task evaluator from the
   hook context if hooks set; otherwise the engine-bound ``self.evaluator``).
6. Reward is written back, gateway session deleted, hook teardown runs.

Eval and training differ only in which hooks they install — the driver
loop in :meth:`_run_single` is identical.
"""

from __future__ import annotations

import asyncio
import logging
import resource
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from tqdm import tqdm

from rllm.eval.types import EvalOutput
from rllm.experimental.engine.trace_converter import compute_step_metrics, trace_record_to_step
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory, run_agent_flow
from rllm.utils import colorful_print
from rllm.workflows.workflow import TerminationReason

if TYPE_CHECKING:
    from rllm_model_gateway.models import TraceRecord

    from rllm.experimental.engine.gateway_manager import GatewayManager
    from rllm.types import AgentFlow, Evaluator
    from rllm.utils.episode_logger import EpisodeLogger

logger = logging.getLogger(__name__)

_MIN_FD_LIMIT = 8192


class EnrichMismatchError(RuntimeError):
    """Raised when gateway traces don't align with the agent's reported steps.

    Indicates a real upstream failure (lost trace, empty token_ids in the vLLM
    response, etc.). process_task_with_retry treats it like any other failure
    and reissues the rollout.
    """


@dataclass
class TaskContext:
    """Per-task state returned by :meth:`TaskHooks.setup`.

    Encapsulates the per-task evaluator (resolved by the hook from a task's
    [verifier] config, or pre-bound by the caller), an optional per-task
    agent flow (so SandboxedAgentFlow's per-task sandbox can't leak across
    parallel rollouts), and a teardown callback that releases any per-task
    resources (sandboxes, temp dirs, ...).
    """

    evaluator: Evaluator
    agent_flow: Any = None  # AgentFlow | None — kept loose for the import-cycle reason below
    teardown: Any = None  # Callable[[], None] | None — kept loose to avoid Callable import loop

    def run_teardown(self) -> None:
        if self.teardown is None:
            return
        try:
            self.teardown()
        except Exception:
            logger.exception("TaskContext.teardown raised; suppressing")


@runtime_checkable
class TaskHooks(Protocol):
    """Per-rollout setup/teardown hook for the engine.

    The engine calls :meth:`setup` before the agent flow runs and
    :meth:`TaskContext.run_teardown` after the evaluator runs (or on failure).
    Eval installs hooks that create a sandbox and resolve a per-task verifier;
    training leaves hooks unset and the engine uses ``self.evaluator``
    directly.
    """

    def setup(self, task: Task, agent_flow: AgentFlow, uid: str) -> TaskContext: ...


def enrich_episode_with_traces(
    episode: Episode,
    traces: list[TraceRecord],
    uid: str,
    task: dict,
    *,
    strict: bool = True,
) -> Episode:
    """Merge gateway traces into agent's lightweight Episode.

    Matching strategy (positional):

    - Traces are ordered chronologically.
    - Walk through trajectories in order, match each step to the next trace
      by position.
    - Create training Steps from traces, preserve rewards/done flags from
      agent Steps.

    When ``strict=True`` (default; training path): empty ``prompt_token_ids``
    or ``completion_token_ids`` raise :class:`EnrichMismatchError` so the
    engine's retry path can reissue the rollout. Token IDs are required for
    loss math, and missing ones from vLLM indicate an upstream failure.

    When ``strict=False`` (eval path against non-vLLM upstreams like the
    LiteLLM proxy or OpenAI/Anthropic directly): empty token IDs are OK —
    the evaluator reads ``model_response`` / ``chat_completions``, which are
    populated regardless of token-ID availability.
    """
    if not traces:
        logger.warning("[%s] No traces found — returning episode without token data", uid)
        # Coerce to the canonical Trajectory/Episode shape so downstream
        # pydantic validators (e.g. TrajectoryGroup.trajectories) accept
        # instances produced by agents that imported from rllm.types.
        return Episode(
            id=episode.id,
            task=episode.task,
            is_correct=episode.is_correct,
            termination_reason=episode.termination_reason,
            trajectories=[t if isinstance(t, Trajectory) else Trajectory(**t.model_dump()) for t in episode.trajectories],
            metrics=episode.metrics,
            metadata=episode.metadata,
            artifacts=episode.artifacts,
        )

    # Convert all traces to training steps
    training_steps = [trace_record_to_step(t) for t in traces]

    # Bad traces (missing or empty token_ids) silently corrupt loss math and
    # shrink GRPO groups; raise on real mismatches so retries can reissue.
    n_agent_steps = sum(len(t.steps) for t in episode.trajectories)
    agent_populates_steps = any(len(t.steps) > 0 for t in episode.trajectories)

    # Common case: vLLM returns an empty body on the final call (e.g. prompt
    # hit max_model_len, or weight-sync disconnect). The agent breaks without
    # recording a Step, leaving N+1 traces vs N agent_steps with the trailing
    # one malformed. Drop the trailing trace rather than burn the whole
    # rollout — at high MAX_TURNS the failure rate would exhaust retries.
    if agent_populates_steps and len(training_steps) > n_agent_steps:
        extra = training_steps[n_agent_steps:]
        extras_all_malformed = all(not s.model_output.prompt_ids or not s.model_output.completion_ids for s in extra)
        if extras_all_malformed:
            logger.warning(
                "[%s] dropping %d trailing malformed trace(s); keeping %d aligned with agent_steps",
                uid,
                len(extra),
                n_agent_steps,
            )
            training_steps = training_steps[:n_agent_steps]

    empty_prompt = sum(1 for s in training_steps if not s.model_output.prompt_ids)
    empty_compl = sum(1 for s in training_steps if not s.model_output.completion_ids)
    # Only enforce step-count parity when the agent actually populates steps.
    # Trajectories with no agent steps absorb remaining traces wholesale
    # (see branch below), and trajectories with steps consume traces 1:1.
    traces_short = agent_populates_steps and len(training_steps) < n_agent_steps
    # Empty token IDs are a hard error only in strict (training) mode.
    # Eval against external providers (OpenAI/Anthropic via LiteLLM proxy)
    # legitimately has empty token IDs and that's fine — the evaluator
    # reads `model_response` / `chat_completions`, not token IDs.
    token_ids_missing = strict and (empty_prompt or empty_compl)
    if traces_short or token_ids_missing:
        raise EnrichMismatchError(f"[{uid}] enrich mismatch: traces={len(training_steps)} agent_steps={n_agent_steps} empty_prompt_ids={empty_prompt} empty_completion_ids={empty_compl}")

    # Build enriched trajectories
    enriched_trajectories: list[Trajectory] = []
    trace_idx = 0

    for traj in episode.trajectories:
        traj_steps: list[Step] = []

        if traj.steps:
            # Match agent steps to traces positionally. The validation above
            # guarantees trace_idx < len(training_steps) for every agent_step
            # when agent_populates_steps is True.
            for agent_step in traj.steps:
                step = training_steps[trace_idx]
                # Preserve reward and done from agent's step
                step.reward = agent_step.reward
                step.done = agent_step.done
                trace_idx += 1
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
    metrics = compute_step_metrics(enriched_trajectories)
    metrics["empty"] = int(len(traces) == 0)
    metrics["steps_collected"] = len(traces)
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
        evaluator: Evaluator | None,
        gateway: GatewayManager,
        model: str,
        n_parallel_tasks: int = 128,
        retry_limit: int = 3,
        raise_on_error: bool = True,
        episode_logger: EpisodeLogger | None = None,
        hooks: TaskHooks | None = None,
    ) -> None:
        if evaluator is None and hooks is None:
            raise ValueError("AgentFlowEngine requires either an `evaluator` (single evaluator, typical training) or `hooks` (per-task evaluator + setup/teardown, typical eval). Both cannot be None.")

        self.agent_flow = agent_flow
        self.evaluator = evaluator
        self.gateway = gateway
        self.model = model
        self.n_parallel_tasks = n_parallel_tasks
        self.retry_limit = retry_limit
        self.raise_on_error = raise_on_error
        self.episode_logger = episode_logger
        self.hooks = hooks
        self.executor = ThreadPoolExecutor(max_workers=n_parallel_tasks)
        self._semaphore = asyncio.Semaphore(n_parallel_tasks)

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
        tasks: list[dict | Task],
        task_ids: list[str] | None = None,
        is_validation: bool = False,
        **kwargs,
    ) -> list[Episode]:
        """Run AgentFlows on a list of tasks; return enriched Episodes.

        ``tasks`` may be raw dicts (training path; the engine wraps each
        in a :class:`Task` internally) or fully-constructed :class:`Task`
        objects (eval path; the engine uses them as-is). When ``task_ids``
        is omitted, fresh UUIDs are assigned.

        See :meth:`_run_single` for the per-rollout lifecycle.
        """
        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        task_id_counter: dict[str, int] = defaultdict(int)
        results: list[Episode | None] = [None] * len(tasks)

        futures = []
        for idx, (task, task_id) in enumerate(zip(tasks, task_ids, strict=True)):
            rollout_idx = task_id_counter[task_id]
            task_id_counter[task_id] += 1
            futures.append(self.process_task_with_retry(task, task_id, rollout_idx, idx, is_validation=is_validation))

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

    async def process_task_with_retry(
        self,
        task: dict | Task,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        is_validation: bool = False,
    ) -> tuple[str, int, int, Episode]:
        """Process a single task with retry logic."""
        # `task_for_episode` is what we stash on the resulting Episode's
        # `task` field for downstream consumers (logger, transform pipeline).
        # Callers historically rely on this being a dict; preserve that.
        task_for_episode = task.metadata if isinstance(task, Task) else task
        async with self._semaphore:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}"
                # Clear any traces from a prior failed attempt so the gateway
                # doesn't mix old attempt's traces with the new attempt's steps
                # (positional match in _enrich_episode would then corrupt data).
                if retry_attempt > 1:
                    try:
                        await self.gateway.adelete_session(uid)
                    except Exception as cleanup_err:
                        logger.warning("[%s] failed to clear prior traces before retry: %s", uid, cleanup_err)
                try:
                    episode = await self._run_single(task, uid, is_validation=is_validation)
                    episode.id = uid
                    episode.task = task_for_episode

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
                    logger.error("[%s] Attempt %d/%d failed: %r (type=%s)", uid, retry_attempt, self.retry_limit, e, type(e).__name__)
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
                            task=task_for_episode,
                            is_correct=False,
                            termination_reason=TerminationReason.ERROR,
                            metadata={"error": {"message": str(e)}},
                        ),
                    )

            # Should not reach here, but satisfy type checker
            raise RuntimeError(f"[{uid}] Exhausted all retries")

    async def _run_single(self, task: dict | Task, uid: str, is_validation: bool = False) -> Episode:
        """Run a single AgentFlow rollout end-to-end.

        Lifecycle (identical for training and eval):

        1. ``hooks.setup`` runs (eval: build sandbox + resolve verifier).
        2. Gateway session is created.
        3. Agent flow runs against the gateway session URL.
        4. Traces fetched and Episode enriched with token-level Steps.
        5. Evaluator scores the enriched Episode (per-task from hook context
           if hooks set; else the engine-bound ``self.evaluator``).
        6. Reward + signals written back, gateway session deleted, hook
           teardown runs (in ``finally``).
        """
        loop = asyncio.get_event_loop()
        from pathlib import Path

        # Normalize the task: callers may pass either a raw dict (training
        # path) or a fully-constructed Task (eval path). We keep both
        # representations because the original dict (when present) is what
        # downstream code uses for `episode.task` and what dict-style
        # evaluators expect to read from.
        if isinstance(task, Task):
            task_obj = task
            task_dict = task.metadata
        else:
            task_dict = task
            task_obj = Task(
                id=str(uid),
                instruction=str(task.get("question", task.get("instruction", ""))),
                metadata=task,
                dataset_dir=Path("."),
            )

        # Hook setup (eval: sandbox + per-task verifier resolution)
        ctx: TaskContext | None = None
        if self.hooks is not None:
            ctx = self.hooks.setup(task_obj, self.agent_flow, uid)

        try:
            # 1. Create gateway session
            await self.gateway.acreate_session(uid, is_validation=is_validation)
            try:
                session_url = self.gateway.get_session_url(uid)

                # 2. Build config
                config = AgentConfig(
                    base_url=session_url,
                    model=self.model,
                    session_uid=uid,
                    is_validation=is_validation,
                )

                # 3. Run agent flow (prefers arun if available, else run in executor).
                # The hook may return a per-task agent flow instance (eg. a
                # SandboxedAgentFlow with the task's sandbox already injected);
                # use it when present so parallel tasks don't share mutable
                # ``self._sandbox`` state on the engine-bound ``self.agent_flow``.
                flow_for_task = ctx.agent_flow if (ctx is not None and ctx.agent_flow is not None) else self.agent_flow
                logger.debug("[%s] Starting agent flow at %s", uid, session_url)
                episode = await run_agent_flow(flow_for_task, task_obj, config, executor=self.executor)
                logger.debug("[%s] Agent flow completed, %d trajectories", uid, len(episode.trajectories))

                # 4. Retrieve traces from gateway and enrich episode with token data.
                # Eval (hooks set) doesn't require vLLM-style token IDs because
                # the upstream may be any OpenAI-compatible endpoint; training
                # (no hooks) needs them for loss math and treats missing IDs
                # as a retry-worthy transient failure.
                traces = await self.gateway.aget_traces(uid)
                enriched = enrich_episode_with_traces(
                    episode,
                    traces,
                    uid,
                    task_dict,
                    strict=self.hooks is None,
                )

                # 5. Evaluate the enriched Episode.
                # Per-task evaluator from the hook context wins over the
                # engine-bound evaluator. Hook-resolved evaluators receive
                # the Task object; the engine-bound evaluator receives the
                # raw task dict (training compatibility).
                if ctx is not None:
                    eval_output: EvalOutput = await loop.run_in_executor(
                        self.executor,
                        ctx.evaluator.evaluate,
                        task_obj,
                        enriched,
                    )
                else:
                    assert self.evaluator is not None  # __init__ guarantees one of evaluator/hooks
                    eval_output = await loop.run_in_executor(
                        self.executor,
                        self.evaluator.evaluate,
                        task_dict,
                        enriched,
                    )

                # Apply reward to trajectories that don't already have one.
                # Evaluators for multi-trajectory flows (e.g. solver-judge) may set
                # per-trajectory rewards directly on the episode; those are preserved.
                for traj in enriched.trajectories:
                    if traj.reward is None:
                        traj.reward = eval_output.reward
                    if not traj.signals:
                        traj.signals = {s.name: s.value for s in eval_output.signals}
                enriched.is_correct = eval_output.is_correct

                # Attach eval metrics
                enriched.metrics.update(eval_output.metadata)
                for signal in eval_output.signals:
                    enriched.metrics[signal.name] = signal.value

                if enriched.termination_reason is None:
                    enriched.termination_reason = TerminationReason.ENV_DONE
                return enriched
            finally:
                # 6. Delete traces from gateway DB to prevent unbounded growth
                try:
                    await self.gateway.adelete_session(uid)
                except Exception:
                    logger.exception("[%s] gateway session delete failed; continuing", uid)
        finally:
            if ctx is not None:
                ctx.run_teardown()

    def shutdown(self) -> None:
        """Shutdown the engine and cleanup resources."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
