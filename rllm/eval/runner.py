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

import logging
from typing import TYPE_CHECKING

from rllm.eval.results import EvalItem, EvalResult
from rllm.hooks import FixedEvaluation, SandboxTaskHooks
from rllm.types import AgentFlow, Evaluator
from rllm.workflows.workflow import TerminationReason

if TYPE_CHECKING:
    from rllm.gateway.manager import GatewayManager

logger = logging.getLogger(__name__)


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
    from rllm.gateway.manager import EvalGatewayManager
    from rllm.gateway.tunnel import is_local_sandbox_backend

    # pass@k: expose each task as `attempts` adjacent rollouts while reusing
    # the same Task objects.
    if attempts > 1:
        tasks = [task for task in tasks for _ in range(attempts)]

    # Cap concurrency by the agent flow's hint, if any. The engine's
    # internal semaphore enforces this on the rollout side.
    effective_concurrency = concurrency
    if hasattr(agent_flow, "max_concurrent"):
        effective_concurrency = min(effective_concurrency, agent_flow.max_concurrent)

    # Lifecycle: if the caller gave us a gateway, use it; otherwise build
    # and tear down one ourselves (single-shot).
    owned_gateway = gateway is None
    if owned_gateway:
        # Auto-tunnel for off-host sandboxes (same predicate AgentTrainer uses).
        gateway_tunnel = None if is_local_sandbox_backend(sandbox_backend) else "cloudflared"
        gateway = EvalGatewayManager(upstream_url=base_url, model=model, tunnel=gateway_tunnel)
        gateway.start()

    hooks = SandboxTaskHooks(evaluation=FixedEvaluation(evaluator) if evaluator is not None else None, sandbox_backend=sandbox_backend, use_snapshot=use_snapshot)

    engine = AgentFlowEngine(
        agent_flow=agent_flow,
        evaluator=None,  # hooks resolve the per-task evaluator
        gateway=gateway,
        model=model,
        n_parallel_tasks=effective_concurrency,
        retry_limit=1,  # eval doesn't retry on flow errors
        raise_on_error=False,  # capture per-task errors as error Episodes
        hooks=hooks,
        val_sampling_params=sampling_params or None,  # eval is always validation
    )

    warm_queue = None
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
        task_ids = [getattr(t, "id", None) or str(idx) for idx, t in enumerate(tasks)]
        episodes = await engine.execute_tasks(tasks, task_ids=task_ids, is_validation=True)
    finally:
        if warm_queue is not None:
            warm_queue.shutdown()
        engine.shutdown()
        if owned_gateway:
            try:
                gateway.stop()
            except Exception:
                logger.exception("gateway.stop() raised; suppressing")

    # Aggregate per-rollout EvalItems for the report; with attempts > 1 the
    # expanded index folds back to (task index, attempt).
    items: list[EvalItem] = []
    surviving_episodes: list = []
    for idx, episode in enumerate(episodes):
        task_idx, attempt = divmod(idx, attempts) if attempts > 1 else (idx, 0)
        if episode is None:
            items.append(EvalItem(idx=task_idx, attempt=attempt, reward=0.0, is_correct=False, error="missing episode"))
            continue

        error_msg = None
        if episode.termination_reason == TerminationReason.ERROR:
            err = (episode.metadata or {}).get("error") or {}
            error_msg = err.get("message") if isinstance(err, dict) else str(err)

        signals: dict[str, float] = {}
        if episode.trajectories:
            signals = dict(episode.trajectories[0].signals or {})

        reward = 0.0
        if episode.trajectories and episode.trajectories[0].reward is not None:
            reward = float(episode.trajectories[0].reward)

        if on_episode_complete is not None:
            try:
                on_episode_complete(idx, episode)
            except Exception:
                logger.debug("on_episode_complete callback error", exc_info=True)

        items.append(
            EvalItem(
                idx=task_idx,
                attempt=attempt,
                reward=reward,
                is_correct=bool(episode.is_correct),
                signals=signals,
                error=error_msg,
            )
        )
        if error_msg is None:
            surviving_episodes.append(episode)

    return (EvalResult.from_items(dataset_name, model, agent_name, items, attempts=attempts), surviving_episodes)
