"""run_dataset: parallel orchestration over a list of Tasks via Runner.

Each Task flows through :class:`rllm.runner.Runner`, which resolves the
verifier from the Task itself (or from ``evaluator_override``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import Callable

from tqdm.asyncio import tqdm_asyncio

from rllm.eval.results import EvalItem, EvalResult
from rllm.types import AgentConfig, AgentFlow, Evaluator

logger = logging.getLogger(__name__)

_TASKS_JSONL = "tasks.jsonl"
_INSTRUCTION_PREVIEW_LIMIT = 240


class _TaskLifecycleLog:
    """Append-only JSONL recording each task's start time + identity.

    Used by the local viewer (``rllm view``) to show in-flight tasks
    during a long-running eval — by the time the first trace lands in
    ``traces.db`` we already know the task's instruction and idx, so the
    UI can render an entry the moment a task starts rather than waiting
    on the gateway.

    Thread-safe: a single ``threading.Lock`` serializes appends. Disabled
    when ``run_dir`` is None (CLI invocations like notebook scripts that
    don't persist a run directory).
    """

    def __init__(self, run_dir: str | None) -> None:
        self.path: str | None = os.path.join(run_dir, _TASKS_JSONL) if run_dir else None
        self._lock = threading.Lock()
        if self.path is not None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def task_started(self, idx: int, session_id: str, task) -> None:
        if self.path is None:
            return
        instruction = getattr(task, "instruction", "") or ""
        if isinstance(instruction, str) and len(instruction) > _INSTRUCTION_PREVIEW_LIMIT:
            instruction = instruction[: _INSTRUCTION_PREVIEW_LIMIT - 1] + "…"
        record = {
            "idx": idx,
            "session_id": session_id,
            "task_id": getattr(task, "id", None),
            "instruction": instruction,
            "started_at": time.time(),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)


def _stamp_session_in_url(base_url: str, session_uid: str) -> str:
    """Splice ``/sessions/{session_uid}`` into the gateway base URL.

    The rLLM gateway extracts ``session_id`` from a leading
    ``/sessions/{sid}/v1/...`` path. We rewrite the per-task base URL
    so the harness's stock OpenAI client (which knows nothing about
    session metadata) automatically tags every call with the right
    session, and traces land in the shared gateway db with the
    correct ``(run_id, session_id)`` tuple. Examples::

        http://h:p/v1            -> http://h:p/sessions/eval-0/v1
        http://h:p               -> http://h:p/sessions/eval-0/v1
        http://h:p/v1/           -> http://h:p/sessions/eval-0/v1
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/sessions/{session_uid}/v1"


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
    stamp_session_in_url: bool = False,
    run_dir: str | None = None,
    gateway_auth_token: str | None = None,
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
    task_log = _TaskLifecycleLog(run_dir)

    async def run_one(idx: int, task) -> tuple[EvalItem, object | None]:
        async with semaphore:
            # Per-task fresh agent_flow for sandboxed flows so sandbox state doesn't leak
            af = agent_flow.create_instance() if isinstance(agent_flow, SandboxedAgentFlow) else agent_flow
            runner = Runner(
                agent_flow=af,
                sandbox_backend=sandbox_backend,
                evaluator_override=evaluator_override,
            )
            session_uid = f"eval-{idx}"
            task_base_url = _stamp_session_in_url(base_url, session_uid) if stamp_session_in_url else base_url
            metadata: dict = {}
            if gateway_auth_token:
                # Harnesses read this and overload provider key vars
                # (OPENAI_API_KEY, ANTHROPIC_API_KEY, …) to the bearer
                # token. The gateway re-stamps with the real upstream
                # key from server-side env before forwarding.
                metadata["gateway_auth_token"] = gateway_auth_token
            config = AgentConfig(
                base_url=task_base_url,
                model=model,
                session_uid=session_uid,
                metadata=metadata,
            )
            task_log.task_started(idx, session_uid, task)
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
