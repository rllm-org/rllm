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
from rllm.hooks import SandboxTaskHooks
from rllm.types import AgentFlow, Evaluator
from rllm.workflows.workflow import TerminationReason

if TYPE_CHECKING:
    from rllm.experimental.engine.gateway_manager import GatewayManager

logger = logging.getLogger(__name__)


async def run_dataset(
    tasks: list,  # list[rllm.types.Task]
    agent_flow: AgentFlow,
    base_url: str,
    model: str,
    *,
    concurrency: int = 64,
    sandbox_backend: str | None = None,
    agent_name: str = "",
    dataset_name: str = "unknown",
    on_episode_complete=None,
    evaluator_override: Evaluator | None = None,
    gateway: GatewayManager | None = None,
) -> tuple[EvalResult, list]:
    """Run a list of :class:`rllm.types.Task` objects through :class:`AgentFlowEngine`.

    Per-task: the engine creates a gateway session, runs the agent flow
    against the session URL, fetches traces, enriches the Episode, then
    runs the per-task evaluator (or ``evaluator_override`` if set).

    Args:
        gateway: Optional pre-started gateway. When ``None``, this function
            constructs an :class:`EvalGatewayManager` pointing at
            ``base_url`` and tears it down on exit. When provided, the
            caller owns the lifecycle (used by ``rllm.cli.eval`` so the
            gateway can stay up across multiple runs).
        evaluator_override: Bind a single evaluator to all tasks (CLI's
            ``--evaluator`` flag). When ``None``, ``SandboxTaskHooks``
            resolves a per-task verifier from the task's ``[verifier]``
            config.

    Returns ``(EvalResult, list[Episode])``.
    """
    # Lazy imports — both modules pull in `rllm.eval.types` at import time,
    # which loads the parent `rllm.eval` package and creates a circular
    # import (rllm.eval.__init__ → rllm.eval.runner → here). Importing
    # them inside the function breaks the cycle.
    from rllm.engine.agentflow_engine import AgentFlowEngine
    from rllm.experimental.engine.gateway_manager import EvalGatewayManager

    # Cap concurrency by the agent flow's hint, if any. The engine's
    # internal semaphore enforces this on the rollout side.
    effective_concurrency = concurrency
    if hasattr(agent_flow, "max_concurrent"):
        effective_concurrency = min(effective_concurrency, agent_flow.max_concurrent)

    # Lifecycle: if the caller gave us a gateway, use it; otherwise build
    # and tear down one ourselves (single-shot).
    owned_gateway = gateway is None
    if owned_gateway:
        gateway = EvalGatewayManager(upstream_url=base_url, model=model)
        gateway.start()

    hooks = SandboxTaskHooks(evaluator_override=evaluator_override, sandbox_backend=sandbox_backend)

    engine = AgentFlowEngine(
        agent_flow=agent_flow,
        evaluator=None,  # hooks resolve the per-task evaluator
        gateway=gateway,
        model=model,
        n_parallel_tasks=effective_concurrency,
        retry_limit=1,  # eval doesn't retry on flow errors
        raise_on_error=False,  # capture per-task errors as error Episodes
        hooks=hooks,
    )

    try:
        # task_ids carry the original Task.id so GRPO-style grouping (if a
        # downstream consumer wants it) is stable; the engine's session uid
        # becomes f"{task.id}:0" which matches training's convention.
        task_ids = [getattr(t, "id", None) or str(idx) for idx, t in enumerate(tasks)]
        episodes = await engine.execute_tasks(tasks, task_ids=task_ids, is_validation=True)
    finally:
        engine.shutdown()
        if owned_gateway:
            try:
                gateway.stop()
            except Exception:
                logger.exception("gateway.stop() raised; suppressing")

    # Aggregate per-task EvalItems for the report.
    items: list[EvalItem] = []
    surviving_episodes: list = []
    for idx, episode in enumerate(episodes):
        if episode is None:
            items.append(EvalItem(idx=idx, reward=0.0, is_correct=False, error="missing episode"))
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
                idx=idx,
                reward=reward,
                is_correct=bool(episode.is_correct),
                signals=signals,
                error=error_msg,
            )
        )
        if error_msg is None:
            surviving_episodes.append(episode)

    return (EvalResult.from_items(dataset_name, model, agent_name, items), surviving_episodes)
