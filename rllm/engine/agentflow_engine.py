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
                # Preserve agent-side fields (the trace doesn't carry these — it
                # only holds the raw LLM call) -- action, reward, done
                step.action = agent_step.action
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
        train_sampling_params: dict | None = None,
        val_sampling_params: dict | None = None,
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
        self.train_sampling_params = train_sampling_params
        self.val_sampling_params = val_sampling_params

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

        Runs per-task pipelines (flow + trace fetch + enrich + evaluate)
        in parallel, streamed via ``asyncio.as_completed`` so the
        rollout-completed log lines arrive as each task finishes. One
        ``POST /sessions/batch_delete`` at the end of the step cleans
        up the trace store.
        """
        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        task_id_counter: dict[str, int] = defaultdict(int)

        futures = []
        uids: list[str] = []
        for idx, (task, task_id) in enumerate(zip(tasks, task_ids, strict=True)):
            rollout_idx = task_id_counter[task_id]
            task_id_counter[task_id] += 1
            uid = f"{task_id}:{rollout_idx}"
            uids.append(uid)
            futures.append(self.process_task_with_retry(task, task_id, rollout_idx, idx, is_validation=is_validation))

        results: list[Episode | None] = [None] * len(tasks)
        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                task_id, rollout_idx, result_idx, episode = await future
                results[result_idx] = episode
                pbar.update(1)

        ordered_results: list[Episode] = results  # type: ignore[assignment]

        # Batch session delete at end of step to keep the trace store from
        # growing unboundedly. One ``POST /sessions/batch_delete`` for all
        # uids instead of N (flush + DELETE) RTTs.
        if uids:
            try:
                await self.gateway.adelete_sessions(uids)
            except Exception:
                logger.exception("Batch session delete failed; sessions may linger in the trace store")

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
        """Run the full per-task pipeline with retry.

        Each attempt runs flow + trace fetch + enrich + evaluate. On
        retry, stale traces from the prior attempt are cleared first so
        the new attempt's enrich doesn't see a mix of trace records.
        """
        task_for_episode = task.metadata if isinstance(task, Task) else task
        from pathlib import Path

        if isinstance(task, Task):
            task_obj = task
            task_dict = task.metadata
        else:
            task_dict = task
            task_obj = Task(
                id=str(task_id),
                instruction=str(task.get("question", task.get("instruction", ""))),
                metadata=task,
                dataset_dir=Path("."),
            )

        async with self._semaphore:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}"
                if retry_attempt > 1:
                    try:
                        await self.gateway.adelete_session(uid)
                    except Exception as cleanup_err:
                        logger.warning("[%s] failed to clear prior traces before retry: %s", uid, cleanup_err)
                try:
                    episode = await self._run_single(task_obj, task_dict, uid, is_validation=is_validation)
                    episode.id = uid
                    episode.task = task_for_episode

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

            raise RuntimeError(f"[{task_id}:{rollout_idx}] Exhausted all retries")

    async def _run_single(self, task_obj: Task, task_dict: dict, uid: str, is_validation: bool = False) -> Episode:
        """Run one full per-task pipeline: flow → fetch traces → enrich → evaluate."""
        raw_episode, ctx = await self._run_flow_only(
            task_obj=task_obj,
            task_dict=task_dict,
            uid=uid,
            is_validation=is_validation,
        )
        try:
            traces = await self.gateway.aget_traces(uid)
            return await self._finish_episode(
                raw_episode=raw_episode,
                traces=traces,
                uid=uid,
                task_obj=task_obj,
                task_dict=task_dict,
                ctx=ctx,
            )
        finally:
            if ctx is not None:
                ctx.run_teardown()

    async def _run_flow_only(
        self,
        task_obj: Task,
        task_dict: dict,
        uid: str,
        is_validation: bool = False,
    ) -> tuple[Episode, TaskContext | None]:
        """Phase 1: hook setup + agent flow. No trace fetch, no enrich, no evaluate.

        On flow failure, runs ``ctx.run_teardown()`` if a hook context was
        created, then re-raises so :meth:`process_task_with_retry` can retry.
        On success, returns the raw episode plus the hook context (caller is
        responsible for running teardown after enrich+evaluate).
        """
        ctx: TaskContext | None = None
        if self.hooks is not None:
            ctx = self.hooks.setup(task_obj, self.agent_flow, uid)

        try:
            # Fix (1): no acreate_session HTTP call. SessionRoutingMiddleware
            # extracts the session id from the URL path and tolerates unknown
            # sessions; sampling_params already flow into chat.completions
            # via AgentConfig.sampling_params (spread by the agent flow as
            # ``**sampling``), so the per-session params record stored by
            # the server-side ``create_session`` is redundant.
            session_url = self.gateway.get_session_url(uid)

            config = AgentConfig(
                base_url=session_url,
                model=self.model,
                session_uid=uid,
                is_validation=is_validation,
                sampling_params=(self.train_sampling_params if not is_validation else self.val_sampling_params) or {},
            )

            # The hook may return a per-task agent flow instance (eg. a
            # SandboxedAgentFlow with the task's sandbox already injected);
            # use it when present so parallel tasks don't share mutable
            # ``self._sandbox`` state on the engine-bound ``self.agent_flow``.
            flow_for_task = ctx.agent_flow if (ctx is not None and ctx.agent_flow is not None) else self.agent_flow
            logger.debug("[%s] Starting agent flow at %s", uid, session_url)
            episode = await run_agent_flow(flow_for_task, task_obj, config, executor=self.executor)
            logger.debug("[%s] Agent flow completed, %d trajectories", uid, len(episode.trajectories))
            return episode, ctx
        except BaseException:
            # Tear down per-attempt resources on failure; success path
            # defers teardown until after enrich+evaluate completes.
            if ctx is not None:
                try:
                    ctx.run_teardown()
                except Exception:
                    logger.exception("[%s] hook teardown failed during error recovery", uid)
            raise

    async def _finish_episode(
        self,
        raw_episode: Episode,
        traces: list[TraceRecord],
        uid: str,
        task_obj: Task,
        task_dict: dict,
        ctx: TaskContext | None,
    ) -> Episode:
        """Phase 3: enrich raw episode with traces, run evaluator, apply rewards."""
        loop = asyncio.get_event_loop()

        enriched = enrich_episode_with_traces(
            raw_episode,
            traces,
            uid,
            task_dict,
            strict=self.hooks is None,
        )

        # Per-task evaluator from the hook context wins over the engine-bound
        # evaluator. Hook-resolved evaluators receive the Task object; the
        # engine-bound evaluator receives the raw task dict (training compat).
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

        # Evaluators for multi-trajectory flows (e.g. solver-judge) may set
        # per-trajectory rewards directly on the episode; preserve those.
        for traj in enriched.trajectories:
            if traj.reward is None:
                traj.reward = eval_output.reward
            if not traj.signals:
                traj.signals = {s.name: s.value for s in eval_output.signals}
        enriched.is_correct = eval_output.is_correct

        enriched.metrics.update(eval_output.metadata)
        for signal in eval_output.signals:
            enriched.metrics[signal.name] = signal.value

        if enriched.termination_reason is None:
            enriched.termination_reason = TerminationReason.ENV_DONE
        return enriched

    def shutdown(self) -> None:
        """Shutdown the engine and cleanup resources."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
