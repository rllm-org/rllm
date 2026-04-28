"""run_dataset: parallel orchestration over a list of Tasks via Runner.

Each Task flows through :class:`rllm.runner.Runner`, which resolves the
verifier from the Task itself (or from ``evaluator_override``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from tqdm.asyncio import tqdm_asyncio

from rllm.eval.results import EvalItem, EvalResult
from rllm.types import AgentConfig, AgentFlow, Evaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run a list of Tasks through Runner with concurrency
# ---------------------------------------------------------------------------


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
    on_episode_complete: Callable | None = None,
    evaluator_override: Evaluator | None = None,
) -> tuple[EvalResult, list]:
    """Run a list of :class:`rllm.types.Task` objects through :class:`rllm.runner.Runner`.

    Per-task: creates a fresh :class:`Runner`, optionally with a per-task
    copy of the agent_flow (for sandboxed flows), and awaits its result.
    Concurrency is bounded by ``min(concurrency, agent_flow.max_concurrent)``.

    Args:
        evaluator_override: If provided, all tasks are scored with this
            evaluator instead of their per-task verifier (CLI ``--evaluator``).

    Returns ``(EvalResult, list[Episode])``.
    """
    from rllm.runner import Runner
    from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow

    if hasattr(agent_flow, "max_concurrent"):
        concurrency = min(concurrency, agent_flow.max_concurrent)
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(idx: int, task) -> tuple[EvalItem, object | None]:
        async with semaphore:
            # Per-task fresh agent_flow for sandboxed flows so sandbox state doesn't leak
            af = agent_flow.create_instance() if isinstance(agent_flow, SandboxedAgentFlow) else agent_flow
            runner = Runner(
                agent_flow=af,
                sandbox_backend=sandbox_backend,
                evaluator_override=evaluator_override,
            )
            config = AgentConfig(
                base_url=base_url,
                model=model,
                session_uid=f"eval-{idx}",
            )
            try:
                episode = await runner.run(task, config)
                # Pull the first signal map for reporting
                signals: dict[str, float] = {}
                if episode.trajectories:
                    signals = dict(episode.trajectories[0].signals or {})

                # Reward = primary trajectory's reward
                reward = 0.0
                if episode.trajectories:
                    reward = episode.trajectories[0].reward or 0.0

                if on_episode_complete is not None:
                    try:
                        on_episode_complete(idx, episode)
                    except Exception:
                        logger.debug("on_episode_complete callback error", exc_info=True)

                return (
                    EvalItem(
                        idx=idx,
                        reward=reward,
                        is_correct=bool(episode.is_correct),
                        signals=signals,
                    ),
                    episode,
                )
            except Exception as e:
                logger.warning("Error evaluating example %d: %s", idx, e)
                return (EvalItem(idx=idx, reward=0.0, is_correct=False, error=str(e)), None)

    task_coros = [run_one(i, t) for i, t in enumerate(tasks)]
    results = await tqdm_asyncio.gather(*task_coros, desc="Evaluating")
    items = [r[0] for r in results]
    episodes = [r[1] for r in results if r[1] is not None]
    return (EvalResult.from_items(dataset_name, model, agent_name, items), episodes)
