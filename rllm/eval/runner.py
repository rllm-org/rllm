"""run_dataset: drives :class:`AgentFlowEngine` over a list of Tasks for ``rllm eval``.

Eval shares the same execution engine as training. The eval-specific
concerns — per-task verifier resolution and per-task sandbox lifecycle —
are encapsulated in :class:`rllm.hooks.SandboxTaskHooks` and threaded
into the engine via its :class:`TaskHooks` protocol.

The gateway sits in front of every LLM call so flows that ``return None``
(framework-cookbook style) get their Steps populated from gateway-captured
traces, exactly as they do at training time.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rllm.eval.results import EvalItem, EvalResult
from rllm.hooks import FixedEvaluation, SandboxTaskHooks
from rllm.types import INFRA_ERROR_REASONS, AgentFlow, Evaluator

if TYPE_CHECKING:
    from rllm.gateway.manager import GatewayManager

logger = logging.getLogger(__name__)


def _notify_episode_complete(callback, flat_idx: int, task_idx: int, attempt: int, episode) -> None:
    """Call the extended callback while preserving the historical two-arg API."""
    try:
        inspect.signature(callback).bind(flat_idx, task_idx, attempt, episode)
    except (TypeError, ValueError):
        callback(flat_idx, episode)
    else:
        callback(flat_idx, task_idx, attempt, episode)


async def run_dataset(
    tasks: list,  # list[rllm.types.Task]
    agent_flow: AgentFlow,
    base_url: str,
    model: str,
    *,
    concurrency: int = 64,
    sandbox_backend: str | None = None,
    use_snapshot: bool = True,
    warm_queue_size: int = 0,
    agent_name: str = "",
    dataset_name: str = "unknown",
    on_episode_complete=None,
    evaluator: Evaluator | None = None,
    gateway: GatewayManager | None = None,
    sampling_params: dict | None = None,
    attempts: int = 1,
    run_dir: str | Path | None = None,
    resume_items: list[EvalItem] | None = None,
) -> tuple[EvalResult, list]:
    """Run a list of :class:`rllm.types.Task` objects through :class:`AgentFlowEngine`.

    Per-task: the engine creates a gateway session, runs the agent flow
    against the session URL, fetches traces, enriches the Episode, then
    runs the per-task evaluator (or the fixed ``evaluator`` if set).

    Args:
        gateway: Optional pre-started gateway. When ``None``, this function
            constructs an :class:`EvalGatewayManager` pointing at
            ``base_url`` and tears it down on exit. When provided, the
            caller owns the lifecycle (used by ``rllm.cli.eval`` so the
            gateway can stay up across multiple runs).
        evaluator: Bind a single evaluator to all tasks (CLI's
            ``--evaluator`` flag; the hooks' ``FixedEvaluation`` policy).
            When ``None``, ``SandboxTaskHooks`` resolves a per-task verifier
            from the task's ``[verifier]`` config.
        sampling_params: Resolved sampling params from the CLI, attached to each
            gateway session so the gateway enforces them on every LLM call. ``None``
            or empty → flows/harnesses keep their own params.
        attempts: Independent rollouts per task (pass@k). Each task is expanded
            into ``attempts`` adjacent copies; the engine numbers sibling rollouts
            ``task_id:0..n-1`` (training's GRPO convention) and the EvalResult
            groups them back by task to compute ``pass_at``.

    Returns ``(EvalResult, list[Episode])``.
    """
    # Lazy imports — both modules pull in `rllm.eval.types` at import time,
    # which loads the parent `rllm.eval` package and creates a circular
    # import (rllm.eval.__init__ → rllm.eval.runner → here). Importing
    # them inside the function breaks the cycle.
    from rllm.engine.agentflow_engine import AgentFlowEngine
    from rllm.eval.lifecycle import EvalRunContext, call_eval_lifecycle
    from rllm.gateway.manager import EvalGatewayManager
    from rllm.gateway.tunnel import is_local_sandbox_backend

    original_tasks = list(tasks)
    full_rollout_map = [(task_idx, attempt) for task_idx in range(len(original_tasks)) for attempt in range(attempts)]
    valid_rollouts = set(full_rollout_map)
    seen_resume_rollouts: set[tuple[int, int]] = set()
    for item in resume_items or []:
        key = (item.idx, item.attempt)
        if key not in valid_rollouts:
            raise ValueError(f"Resume item is outside this run manifest: task={item.idx}, attempt={item.attempt}")
        if key in seen_resume_rollouts:
            raise ValueError(f"Duplicate resume item: task={item.idx}, attempt={item.attempt}")
        seen_resume_rollouts.add(key)
        expected_task_id = str(getattr(original_tasks[item.idx], "id", None) or item.idx)
        if item.task_id is not None and item.task_id != expected_task_id:
            raise ValueError(f"Resume task ID mismatch at index {item.idx}: expected {expected_task_id}, got {item.task_id}")
    successful_resume_items = [item for item in (resume_items or []) if item.error is None]
    rollout_map = list(full_rollout_map)
    if resume_items:
        completed = {(item.idx, item.attempt) for item in successful_resume_items}
        rollout_map = [key for key in rollout_map if key not in completed]
        tasks = [original_tasks[task_idx] for task_idx, _attempt in rollout_map]
    elif attempts > 1:
        tasks = [task for task in original_tasks for _ in range(attempts)]

    if not rollout_map:
        items = sorted(successful_resume_items, key=lambda item: (item.idx, item.attempt))
        return EvalResult.from_items(dataset_name, model, agent_name, items, attempts=attempts), []

    # Cap concurrency by the agent flow's hint, if any. The engine's
    # internal semaphore enforces this on the rollout side.
    effective_concurrency = concurrency
    if hasattr(agent_flow, "max_concurrent"):
        effective_concurrency = min(effective_concurrency, agent_flow.max_concurrent)

    # Lifecycle: if the caller gave us a gateway, use it; otherwise build
    # and tear down one ourselves (single-shot).
    owned_gateway = gateway is None
    if owned_gateway:
        # Each remote eval owns its tunnel instead of reusing the singleton
        # `rllm tunnel up` daemon and its fixed origin port.
        gateway_tunnel: str | None = None
        gateway_port: int | None = None
        if not is_local_sandbox_backend(sandbox_backend):
            from rllm.gateway.tunnel import resolve_auto_tunnel

            gateway_tunnel, tunnel_warning = resolve_auto_tunnel()
            if tunnel_warning:
                logger.warning(tunnel_warning)
            if gateway_tunnel.startswith(("http://", "https://")):
                # A tunnel URL means an already-running forwarder; the gateway
                # must bind wherever it forwards (a free-port pick would leave
                # the tunnel pointing at nothing). The daemon's recorded
                # ``upstream`` is authoritative; a URL supplied some other way
                # (env var) falls back to the setup config's port.
                from urllib.parse import urlparse

                from rllm.gateway.tunnel import live_tunnel

                state = live_tunnel() or {}
                upstream = state.get("upstream") if state.get("url") == gateway_tunnel else None
                gateway_port = urlparse(upstream).port if upstream else None
                if gateway_port is None:
                    from rllm.eval.config import load_tunnel_config

                    gateway_port = int(load_tunnel_config().get("port") or 9090)
                    logger.warning(
                        "Tunnel URL %s has no matching daemon state; binding the gateway to port %d from the tunnel config — it must match the port that URL forwards to.",
                        gateway_tunnel,
                        gateway_port,
                    )
        gateway = EvalGatewayManager(upstream_url=base_url, model=model, tunnel=gateway_tunnel, port=gateway_port)
        gateway.start()

    lifecycle_context = EvalRunContext(
        dataset_name=dataset_name,
        agent_name=agent_name,
        model=model,
        base_url=base_url,
        task_count=len(original_tasks) * attempts,
        concurrency=effective_concurrency,
        attempts=attempts,
        sampling_params=dict(sampling_params or {}),
        run_dir=Path(run_dir).expanduser() if run_dir is not None else None,
        tasks=tuple(tasks),
    )
    try:
        await call_eval_lifecycle(agent_flow, "prepare_eval", lifecycle_context)
    except BaseException as exc:
        try:
            await call_eval_lifecycle(agent_flow, "finalize_eval", lifecycle_context, exc)
        finally:
            if owned_gateway:
                gateway.stop()
        raise

    hooks = SandboxTaskHooks(evaluation=FixedEvaluation(evaluator) if evaluator is not None else None, sandbox_backend=sandbox_backend, use_snapshot=use_snapshot)

    engine = AgentFlowEngine(
        agent_flow=agent_flow,
        evaluator=None,  # hooks resolve the per-task evaluator
        gateway=gateway,
        model=model,
        n_parallel_tasks=effective_concurrency,
        # One retry: rollout errors are usually transient infra (sandbox reaped,
        # flaky create, install blip), not flow bugs. Without it they become
        # permanent zeros that depress the score; only errored tasks re-run.
        retry_limit=2,
        raise_on_error=False,  # capture per-task errors as error Episodes
        hooks=hooks,
        val_sampling_params=sampling_params or None,  # eval is always validation
    )

    warm_queue = None
    run_error: BaseException | None = None
    try:
        # Warm queue: prefetch this run's next sandboxes ahead of consumption.
        # Negative size means "match concurrency"; it only helps when sandboxes
        # are actually created, so gate on a chosen sandbox backend.
        if warm_queue_size != 0 and sandbox_backend:
            from rllm.sandbox.snapshot import install_script_for
            from rllm.sandbox.warm_queue import WarmQueue

            size = effective_concurrency if warm_queue_size < 0 else warm_queue_size
            warm_queue = WarmQueue(list(tasks), sandbox_backend, hooks.registry, size, install_script=install_script_for(agent_flow))
            hooks.warm_queue = warm_queue
            warm_queue.start()

        # task_ids carry the original Task.id so GRPO-style grouping (if a
        # downstream consumer wants it) is stable; the engine's session uid
        # becomes f"{task.id}:0" which matches training's convention.
        if resume_items:
            task_ids = [f"{getattr(task, 'id', None) or task_idx}~attempt-{attempt}" for task, (task_idx, attempt) in zip(tasks, rollout_map, strict=True)]
        else:
            task_ids = [getattr(t, "id", None) or str(idx) for idx, t in enumerate(tasks)]

        def _stream_episode(result_idx, episode):
            if on_episode_complete is None:
                return
            task_idx, attempt = rollout_map[result_idx]
            _notify_episode_complete(on_episode_complete, task_idx * attempts + attempt, task_idx, attempt, episode)

        episodes = await engine.execute_tasks(
            tasks,
            task_ids=task_ids,
            is_validation=True,
            on_episode_complete=_stream_episode if on_episode_complete is not None else None,
        )
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        if warm_queue is not None:
            warm_queue.shutdown()
        engine.shutdown()
        if owned_gateway:
            try:
                gateway.stop()
            except Exception:
                logger.exception("gateway.stop() raised; suppressing")
        try:
            await call_eval_lifecycle(agent_flow, "finalize_eval", lifecycle_context, run_error)
        except Exception:
            if run_error is None:
                raise
            logger.exception("agent finalize_eval raised while handling another error; suppressing")

    # Aggregate per-rollout EvalItems for the report; with attempts > 1 the
    # expanded index folds back to (task index, attempt).
    items: list[EvalItem] = list(successful_resume_items)
    surviving_episodes: list = []
    for idx, episode in enumerate(episodes):
        task_idx, attempt = rollout_map[idx]
        task_id = str(getattr(original_tasks[task_idx], "id", None) or task_idx)
        if episode is None:
            items.append(EvalItem(idx=task_idx, attempt=attempt, task_id=task_id, reward=0.0, is_correct=False, error="missing episode"))
            continue

        # An infra/grading failure (sandbox/setup/verifier/grading) means the
        # reward isn't a real task score — surface it as an error so it's counted
        # separately from genuine task failures. An agent TIMEOUT is NOT an error
        # here: it's graded on partial state, so its reward stands.
        reason = episode.termination_reason
        error_msg = None
        if reason in INFRA_ERROR_REASONS:
            err = (episode.metadata or {}).get("error") or {}
            if isinstance(err, dict):
                error_msg = err.get("error_type") or err.get("message") or reason.value
            else:
                error_msg = str(err) or reason.value

        signals: dict[str, float] = {}
        if episode.trajectories:
            signals = dict(episode.trajectories[0].signals or {})

        reward = 0.0
        if episode.trajectories and episode.trajectories[0].reward is not None:
            reward = float(episode.trajectories[0].reward)

        # NOTE: on_episode_complete is now invoked *streaming* inside
        # engine.execute_tasks (as each rollout finishes), not here — so UI
        # uploads + local writes happen progressively instead of in a burst.

        items.append(
            EvalItem(
                idx=task_idx,
                attempt=attempt,
                task_id=task_id,
                reward=reward,
                is_correct=bool(episode.is_correct),
                signals=signals,
                error=error_msg,
                termination_reason=reason.value if reason is not None else None,
            )
        )
        if error_msg is None:
            surviving_episodes.append(episode)

    items.sort(key=lambda item: (item.idx, item.attempt))
    return (EvalResult.from_items(dataset_name, model, agent_name, items, attempts=attempts), surviving_episodes)
