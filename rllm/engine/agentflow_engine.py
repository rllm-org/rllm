"""AgentFlowEngine: runs AgentFlows with gateway-mediated trace capture.

Single execution engine for both training and eval. Each rollout:

1. ``hooks.setup(task, agent_flow, uid)`` runs — sandbox-style hooks create
   a per-task sandbox + resolve a per-task verifier; a bare ``evaluator=``
   is wrapped in :class:`rllm.hooks.FixedEvaluatorHooks` so the engine has
   exactly one execution path.
2. The agent flow runs against the gateway session URL.
3. Traces are fetched and the Episode is enriched with token-level Steps
   (strict for training, relaxed for validation).
4. The hook-resolved evaluator scores the enriched Episode.
5. Reward is written back; the hook context is torn down. Sessions are
   batch-deleted from the trace store at the end of the step.

Eval and training differ only in which hooks they install — the per-task
pipeline in :meth:`_run_single` is identical.
"""

from __future__ import annotations

import asyncio
import logging
import resource
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from tqdm import tqdm

from rllm.data.utils import task_from_row
from rllm.engine.trace_converter import compute_step_metrics, trace_record_to_step
from rllm.eval.types import EvalOutput
from rllm.gateway.manager import container_reachable_url
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory, flow_accepts_env, run_agent_flow
from rllm.utils import colorful_print
from rllm.workflows.workflow import TerminationReason

if TYPE_CHECKING:
    from rllm_model_gateway.models import TraceRecord

    from rllm.gateway.manager import GatewayManager
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
    [verifier] config, or pre-bound by the caller), the task's live sandbox
    (``env``, ``None`` for host-only rollouts) with the backend that
    provisioned it, and a teardown callback that releases any per-task
    resources (sandboxes, temp dirs, ...).
    """

    evaluator: Evaluator
    env: Any = None  # Sandbox | None — kept loose for the import-cycle reason below
    env_backend: str | None = None  # backend that actually provisioned env
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
    Sandbox-style hooks create a sandbox and resolve a per-task verifier;
    :class:`rllm.hooks.FixedEvaluatorHooks` binds one evaluator to every task
    and provisions nothing.
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


def _summarize_llm_latencies(traces: list[Any], agentflow_s: float) -> tuple[float, float]:
    """Return ``(llm_sum_s, llm_wall_s)`` from trace latencies (sum and interval-union)."""
    if not traces:
        return 0.0, 0.0

    llm_sum_s = sum(getattr(tr, "latency_ms", 0.0) or 0.0 for tr in traces) / 1000.0

    intervals: list[tuple[float, float]] = []
    for tr in traces:
        end = float(getattr(tr, "timestamp", 0.0) or 0.0)
        dur = (getattr(tr, "latency_ms", 0.0) or 0.0) / 1000.0
        if end > 0 and dur > 0:
            intervals.append((end - dur, end))
    if not intervals:
        return llm_sum_s, min(llm_sum_s, agentflow_s)

    intervals.sort()
    merged_total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged_total += cur_end - cur_start
            cur_start, cur_end = start, end
    merged_total += cur_end - cur_start
    return llm_sum_s, min(merged_total, agentflow_s) if agentflow_s > 0 else merged_total


_TIMING_PHASES_DISPLAY: tuple[tuple[str, str], ...] = (
    ("setup", "time/setup_s"),
    ("agentflow", "time/agentflow_s"),
    ("evaluator", "time/evaluator_s"),
    ("teardown", "time/teardown_s"),
)


def _format_timing_breakdown(metrics: dict[str, float]) -> str:
    """Compact per-rollout timing summary, e.g. ``setup=16s agentflow=1162s [llm=1100s/15 steps ||1.4x] evaluator=9s teardown=0s``.

    The ``agentflow`` phase is annotated with ``[llm=Xs/N steps]`` (wall-clock
    LLM wait, interval-union), plus ``||N.Nx`` when parallel LLM calls push
    the sum past ``agentflow_s``. Empty when no timings present.
    """
    total = metrics.get("time/rollout_s")
    if total is None:
        return ""
    parts: list[str] = []
    for label, key in _TIMING_PHASES_DISPLAY:
        if key not in metrics:
            continue
        if label == "agentflow":
            llm_wall = metrics.get("time/agentflow_llm_wall_s")
            llm_sum = metrics.get("time/agentflow_llm_sum_s")
            n_turns = metrics.get("n_turns")
            agentflow_s = metrics[key]
            if llm_wall is not None and n_turns is not None and n_turns > 0:
                step_label = "step" if int(n_turns) == 1 else "steps"
                pieces = [f"llm={llm_wall:.0f}s/{int(n_turns)} {step_label}"]
                if llm_sum is not None and agentflow_s > 0 and llm_sum > agentflow_s * 1.05:
                    pieces.append(f"||{llm_sum / agentflow_s:.1f}x")
                parts.append(f"agentflow={agentflow_s:.0f}s [{' '.join(pieces)}]")
            else:
                parts.append(f"agentflow={agentflow_s:.0f}s")
        else:
            parts.append(f"{label}={metrics[key]:.0f}s")
    inner = f" ({' '.join(parts)})" if parts else ""
    return f" in {total:.0f}s{inner}"


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
            raise ValueError("AgentFlowEngine requires either an `evaluator` (single evaluator for every task) or `hooks` (per-task evaluator + setup/teardown). Both cannot be None.")
        if hooks is None:
            from rllm.hooks import FixedEvaluatorHooks

            hooks = FixedEvaluatorHooks(evaluator)

        self._flow_accepts_env = flow_accepts_env(agent_flow)
        if getattr(agent_flow, "needs_env", False) and not self._flow_accepts_env:
            raise TypeError(f"{type(agent_flow).__name__} declares needs_env but its run/arun has no keyword-only 'env' parameter; declare run(self, task, config, *, env).")

        self.agent_flow = agent_flow
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
        task_obj = task if isinstance(task, Task) else task_from_row(task, task_id)

        async with self._semaphore:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}"
                if retry_attempt > 1:
                    try:
                        await self.gateway.adelete_session(uid)
                    except Exception as cleanup_err:
                        logger.warning("[%s] failed to clear prior traces before retry: %s", uid, cleanup_err)
                try:
                    episode = await self._run_single(task_obj, uid, is_validation=is_validation)
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

                    timing_str = _format_timing_breakdown(episode.metrics)
                    colorful_print(
                        f"[{uid}] Rollout completed. Rewards: [{', '.join(reward_strs)}]{timing_str}, Termination: {episode.termination_reason}",
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

    async def _run_single(self, task_obj: Task, uid: str, is_validation: bool = False) -> Episode:
        """Run one full per-task pipeline: flow → fetch traces → enrich → evaluate.

        Records ``time/<phase>_s`` for setup/agentflow/traces/evaluator/
        teardown/rollout into ``episode.metrics``, plus
        ``time/agentflow_llm_wall_s`` (interval-union),
        ``time/agentflow_llm_sum_s`` (naive sum), and ``n_turns``.
        """
        loop = asyncio.get_event_loop()
        timings: dict[str, float] = {}
        rollout_start = time.perf_counter()
        result_holder: dict[str, Episode] = {}

        raw_episode, ctx = await self._run_flow_only(
            task_obj=task_obj,
            uid=uid,
            is_validation=is_validation,
            _timings=timings,
        )
        try:
            t = time.perf_counter()
            traces = await self.gateway.aget_traces(uid)
            timings["time/traces_s"] = time.perf_counter() - t

            enriched = await self._finish_episode(
                raw_episode=raw_episode,
                traces=traces,
                uid=uid,
                task_obj=task_obj,
                ctx=ctx,
                is_validation=is_validation,
                _timings=timings,
            )
            enriched.metrics.update(timings)
            result_holder["episode"] = enriched
            return enriched
        finally:
            # Offload Modal's blocking terminate()/detach() to the executor.
            t = time.perf_counter()
            try:
                await loop.run_in_executor(self.executor, ctx.run_teardown)
            except Exception:
                logger.exception("[%s] task teardown failed; continuing", uid)
            timings["time/teardown_s"] = time.perf_counter() - t
            timings["time/rollout_s"] = time.perf_counter() - rollout_start
            ep = result_holder.get("episode")
            if ep is not None:
                ep.metrics.update(timings)

    async def _run_flow_only(
        self,
        task_obj: Task,
        uid: str,
        is_validation: bool = False,
        _timings: dict[str, float] | None = None,
    ) -> tuple[Episode, TaskContext]:
        """Run hook setup + the agent flow. Returns ``(raw_episode, ctx)``.

        On flow failure, tears down ``ctx`` and re-raises. On success, the
        caller owns ``ctx.run_teardown()``. Records ``time/setup_s`` and
        ``time/agentflow_s`` when ``_timings`` is provided.
        """
        loop = asyncio.get_event_loop()
        if _timings is None:
            _timings = {}

        # Offload hook setup (blocking Modal/docker I/O) to the executor.
        t = time.perf_counter()
        ctx: TaskContext = await loop.run_in_executor(
            self.executor,
            self.hooks.setup,
            task_obj,
            self.agent_flow,
            uid,
        )
        _timings["time/setup_s"] = time.perf_counter() - t

        try:
            if getattr(self.agent_flow, "needs_env", False) and ctx.env is None:
                raise RuntimeError(
                    f"{type(self.agent_flow).__name__} needs a sandbox but hooks {type(self.hooks).__name__} provisioned none — pass hooks=SandboxTaskHooks(...) or run via AgentTrainer / run_dataset."
                )

            # Attach resolved sampling params to the session so the gateway
            # enforces them on every LLM call; skip when there are none.
            session_sampling_params = (self.val_sampling_params if is_validation else self.train_sampling_params) or None
            if session_sampling_params:
                await self.gateway.acreate_session(uid, is_validation=is_validation, sampling_params=session_sampling_params)

            # Flows whose LLM client runs *inside* the env (CLI harnesses)
            # need the publicly-reachable URL — rewritten for in-container
            # networking on the backend that actually provisioned this task's
            # sandbox. Host-side flows keep the local gateway URL so they
            # never depend on a tunnel hostname.
            llm_inside_env = getattr(self.agent_flow, "llm_inside_env", False)
            session_url = self.gateway.get_session_url(uid, public=llm_inside_env)
            if llm_inside_env and ctx.env is not None:
                session_url = container_reachable_url(session_url, ctx.env_backend)

            config = AgentConfig(
                base_url=session_url,
                model=self.model,
                session_uid=uid,
                is_validation=is_validation,
                sampling_params=session_sampling_params or {},
            )
            logger.debug("[%s] Starting agent flow at %s", uid, session_url)
            t = time.perf_counter()
            episode = await run_agent_flow(self.agent_flow, task_obj, config, executor=self.executor, env=ctx.env if self._flow_accepts_env else None)
            _timings["time/agentflow_s"] = time.perf_counter() - t
            logger.debug("[%s] Agent flow completed, %d trajectories", uid, len(episode.trajectories))
            return episode, ctx
        except BaseException:
            # Tear down on failure; success path defers teardown to the caller.
            try:
                await loop.run_in_executor(self.executor, ctx.run_teardown)
            except Exception:
                logger.exception("[%s] hook teardown failed during error recovery", uid)
            raise

    async def _finish_episode(
        self,
        raw_episode: Episode,
        traces: list[TraceRecord],
        uid: str,
        task_obj: Task,
        ctx: TaskContext,
        is_validation: bool = False,
        _timings: dict[str, float] | None = None,
    ) -> Episode:
        """Enrich the raw episode with traces, run the evaluator, apply rewards.

        Training rollouts enrich strictly (empty token IDs raise so the retry
        path reissues — they're required for loss math); validation relaxes
        (non-vLLM upstreams legitimately return no token IDs and evaluators
        read message text). Records ``time/evaluator_s``,
        ``time/agentflow_llm_{sum,wall}_s``, and ``n_turns`` when ``_timings``
        is provided.
        """
        loop = asyncio.get_event_loop()

        enriched = enrich_episode_with_traces(
            raw_episode,
            traces,
            uid,
            task_obj.metadata,
            strict=not is_validation,
        )

        # The hook-resolved evaluator always receives the Task (legacy
        # dict-style evaluators are adapted at hook-construction time).
        t = time.perf_counter()
        eval_output: EvalOutput = await loop.run_in_executor(
            self.executor,
            ctx.evaluator.evaluate,
            task_obj,
            enriched,
        )
        if _timings is not None:
            _timings["time/evaluator_s"] = time.perf_counter() - t
            _agentflow_s = _timings.get("time/agentflow_s", 0.0)
            _llm_sum_s, _llm_wall_s = _summarize_llm_latencies(traces, _agentflow_s)
            _timings["time/agentflow_llm_sum_s"] = _llm_sum_s
            _timings["time/agentflow_llm_wall_s"] = _llm_wall_s
            _timings["n_turns"] = float(len(traces))

        # Preserve per-trajectory rewards set by multi-trajectory evaluators.
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
