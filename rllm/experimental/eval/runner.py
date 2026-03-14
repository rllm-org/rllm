"""EvalRunner: orchestrates parallel evaluation of a model on a dataset using an agent scaffold."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from tqdm.asyncio import tqdm_asyncio

from rllm.experimental.eval.results import EvalItem, EvalResult
from rllm.experimental.eval.task_spec import build_task_spec
from rllm.experimental.eval.types import AgentConfig, AgentFlow, Evaluator, Task, run_agent_flow

logger = logging.getLogger(__name__)


def _is_sandboxed(agent) -> bool:
    """Check if an agent is a SandboxedAgentFlow without importing at module level."""
    from rllm.experimental.agents.sandboxed_agent import SandboxedAgentFlow

    return isinstance(agent, SandboxedAgentFlow)


class EvalRunner:
    """Orchestrates parallel evaluation using a two-stage pipeline.

    Stage 1: AgentFlow.run(task, config) -> Episode (trajectories without rewards)
    Stage 2: Evaluator.evaluate(task, episode) -> EvalOutput (reward + signals)

    The runner writes evaluation results back onto each trajectory and the episode.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        concurrency: int = 64,
        agent_metadata: dict | None = None,
        catalog_entry: dict | None = None,
        benchmark_name: str = "",
    ):
        self.base_url = base_url
        self.model = model
        self.concurrency = concurrency
        self.agent_metadata = agent_metadata or {}
        self.catalog_entry = catalog_entry or {}
        self.benchmark_name = benchmark_name
        self._executor = ThreadPoolExecutor(max_workers=concurrency)

    async def run(self, dataset, agent: AgentFlow, evaluator: Evaluator, agent_name: str = "", on_episode_complete=None) -> tuple[EvalResult, list]:
        """Run evaluation on a dataset using the given agent and evaluator.

        Args:
            dataset: An iterable of task dicts (e.g., rllm.data.Dataset).
            agent: AgentFlow instance with a .run() method.
            evaluator: Evaluator instance with an .evaluate() method.
            agent_name: Name of the agent for reporting.
            on_episode_complete: Optional callback called with each completed episode for progressive logging.

        Returns:
            Tuple of (EvalResult with per-example and aggregate metrics, list of Episodes).
        """
        concurrency = self.concurrency
        if hasattr(agent, "max_concurrent"):
            concurrency = min(concurrency, agent.max_concurrent)
        semaphore = asyncio.Semaphore(concurrency)

        # Build TaskSpec once from catalog + first sample task
        task_spec = None
        if self.catalog_entry:
            sample_task = dataset[0] if len(dataset) > 0 else {}
            task_spec = build_task_spec(self.benchmark_name, self.catalog_entry, sample_task)

        async def eval_one(idx: int, task: dict) -> tuple[EvalItem, object | None]:
            async with semaphore:
                # Create per-task agent instance for sandboxed agents
                task_agent = agent
                is_sandboxed = _is_sandboxed(agent)
                if is_sandboxed:
                    task_agent = agent.create_instance()

                try:
                    metadata = dict(self.agent_metadata)
                    config = AgentConfig(
                        base_url=self.base_url,
                        model=self.model,
                        session_uid=f"eval-{idx}",
                        metadata=metadata,
                    )

                    # Wrap raw task dict into Task object
                    task_obj = Task(data=task, spec=task_spec)

                    # Setup sandbox if needed
                    if is_sandboxed:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(self._executor, task_agent.setup_sandbox, task, config)

                    # Stage 1: Run agent flow (prefers arun if available, else run in executor)
                    episode = await run_agent_flow(task_agent, task_obj, config, executor=self._executor)

                    # Store sandbox reference in artifacts for evaluator access
                    if is_sandboxed and task_agent.sandbox is not None:
                        episode.artifacts["_sandbox"] = task_agent.sandbox

                    # Stage 2: Evaluate (evaluators receive raw dict)
                    eval_output = evaluator.evaluate(task_obj.data, episode)

                    # Write back onto trajectories
                    for traj in episode.trajectories:
                        traj.reward = eval_output.reward
                        traj.signals = {s.name: s.value for s in eval_output.signals}

                    # Set episode-level correctness
                    episode.is_correct = eval_output.is_correct

                    # Clear sandbox artifact before returning (not serializable)
                    episode.artifacts.pop("_sandbox", None)

                    # Notify caller for progressive logging
                    if on_episode_complete is not None:
                        try:
                            on_episode_complete(episode)
                        except Exception:
                            logger.debug("on_episode_complete callback error", exc_info=True)

                    return (
                        EvalItem(
                            idx=idx,
                            reward=eval_output.reward,
                            is_correct=eval_output.is_correct,
                            signals={s.name: s.value for s in eval_output.signals},
                        ),
                        episode,
                    )
                except Exception as e:
                    logger.warning("Error evaluating example %d: %s", idx, e)
                    return (EvalItem(idx=idx, reward=0.0, is_correct=False, error=str(e)), None)
                finally:
                    # Guaranteed sandbox cleanup
                    if is_sandboxed:
                        try:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(self._executor, task_agent.teardown_sandbox)
                        except Exception:
                            logger.exception("Sandbox teardown error for example %d", idx)

        task_coros = [eval_one(i, task) for i, task in enumerate(dataset)]
        results = await tqdm_asyncio.gather(*task_coros, desc="Evaluating")

        items = [r[0] for r in results]
        episodes = [r[1] for r in results if r[1] is not None]

        dataset_name = getattr(dataset, "name", "unknown") or "unknown"
        return (EvalResult.from_items(dataset_name, self.model, agent_name, items), episodes)
