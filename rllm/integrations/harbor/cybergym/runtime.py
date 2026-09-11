"""Thin Harbor runner for CyberGym Level 1 task directories.

Runs the existing Harbor trial (compose main + task-server, agent, post-episode
``tests/test.sh``). Does not reimplement the sidecar. Infra failures raise;
they are not scored as reward 0.0.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rllm.env import env_float
from rllm.integrations.harbor.cybergym.artifacts import score_from_harbor_result
from rllm.integrations.harbor.cybergym.constants import (
    DEFAULT_SCORING_MODE,
    DEFAULT_SESSION_TIMEOUT_SEC,
    SCORING_ANY_OF,
    SCORING_FINAL,
    ScoringMode,
)
from rllm.integrations.harbor.cybergym.detect import is_cybergym_task_dir
from rllm.integrations.harbor.runtime import HarborRuntime
from rllm.integrations.harbor.trial_helper import MODEL_PLACEHOLDER, outcome_to_episode
from rllm.integrations.harbor.utils import resolve_harbor_task_path

logger = logging.getLogger(__name__)

# Harbor compose + sidecar; rLLM must not provision a second agent sandbox.
_DEFAULT_SESSION_TIMEOUT_S = env_float("RLLM_CYBERGYM_SESSION_TIMEOUT_S", DEFAULT_SESSION_TIMEOUT_SEC)


def _resolve_scoring_mode(explicit: str | None) -> ScoringMode:
    raw = (explicit or os.environ.get("RLLM_CYBERGYM_SCORING") or DEFAULT_SCORING_MODE).strip()
    if raw not in {SCORING_ANY_OF, SCORING_FINAL}:
        raise ValueError(f"RLLM_CYBERGYM_SCORING must be {SCORING_ANY_OF!r} or {SCORING_FINAL!r}, got {raw!r}")
    return raw  # type: ignore[return-value]


class CyberGymRuntime(HarborRuntime):
    """Harbor CyberGym Level 1 runtime.

    Same trial as :class:`HarborRuntime`, plus dual-binary artifacts
    (``vul_exit_code``, ``fix_exit_code``, ``n_submissions``) and an explicit
    any-of vs final-submission switch (default Harbor any-of).
    """

    needs_env = False

    def __init__(
        self,
        agent_name: str = "claude-code",
        scoring_mode: ScoringMode | None = None,
        session_timeout: float = _DEFAULT_SESSION_TIMEOUT_S,
        **kwargs: Any,
    ):
        super().__init__(agent_name=agent_name, session_timeout=session_timeout, **kwargs)
        self.scoring_mode = _resolve_scoring_mode(scoring_mode)

    async def arun(self, task, config):
        task_path = resolve_harbor_task_path(task)
        if not is_cybergym_task_dir(task_path):
            logger.warning(
                "CyberGymRuntime: %s is not a Level 1 CyberGym task dir; running as a generic Harbor trial",
                task_path,
            )

        outcome = await self._run_one(
            task_path=task_path,
            model_name=config.model,
            inference_url=config.base_url,
            trial_name=config.session_uid,
        )
        if not outcome.finished:
            raise RuntimeError(f"CyberGym Harbor trial failed ({config.session_uid}): {outcome.error}")

        score = score_from_harbor_result(
            raw_result=outcome.raw_result,
            harbor_reward=outcome.reward,
            trial_uri=outcome.trial_uri,
            scoring_mode=self.scoring_mode,
        )
        episode = outcome_to_episode(outcome, config.session_uid, task.metadata if hasattr(task, "metadata") else task)
        episode.artifacts["harbor_trial_ran"] = True
        episode.artifacts["harbor_reward"] = score.harbor_reward if score.harbor_reward is not None else 0.0
        episode.artifacts["harbor_is_correct"] = bool(score.reward > 0)
        episode.artifacts.update(score.as_artifacts())
        if episode.metrics is not None:
            episode.metrics["reward"] = score.reward
            episode.metrics["is_correct"] = int(score.reward > 0)
            episode.metrics["n_submissions"] = score.n_submissions
            if score.vul_exit_code is not None:
                episode.metrics["vul_exit_code"] = score.vul_exit_code
            if score.fix_exit_code is not None:
                episode.metrics["fix_exit_code"] = score.fix_exit_code
        episode.is_correct = score.reward > 0
        if episode.trajectories:
            episode.trajectories[0].reward = score.reward
        return episode

    async def execute_tasks(self, submissions: list, timeout: float | None = None) -> list:
        from rllm.engine.remote_runtime.protocol import RemoteTaskResult

        if not self._initialized:
            raise RuntimeError("Call initialize() before execute_tasks()")
        if timeout is None:
            timeout = self.session_timeout

        async def _run_submission(sub) -> RemoteTaskResult:
            task_path = resolve_harbor_task_path(sub.task)
            outcome = await self._run_one(
                task_path=task_path,
                model_name=MODEL_PLACEHOLDER,
                inference_url=sub.inference_url,
                trial_name=sub.session_id,
                timeout=timeout,
            )
            if not outcome.finished:
                return RemoteTaskResult(
                    finished=False,
                    session_id=sub.session_id,
                    task_id=sub.task_id,
                    reward=None,
                    error=outcome.error,
                    termination_reason=outcome.termination_reason,
                    elapsed=outcome.elapsed,
                    raw_result=outcome.raw_result,
                    metadata={"trial_uri": outcome.trial_uri} if outcome.trial_uri else {},
                )
            score = score_from_harbor_result(
                raw_result=outcome.raw_result,
                harbor_reward=outcome.reward,
                trial_uri=outcome.trial_uri,
                scoring_mode=self.scoring_mode,
            )
            meta: dict[str, Any] = {
                "vul_exit_code": score.vul_exit_code,
                "fix_exit_code": score.fix_exit_code,
                "n_submissions": score.n_submissions,
                "cybergym_scoring": score.scoring_mode,
            }
            if outcome.trial_uri:
                meta["trial_uri"] = outcome.trial_uri
            return RemoteTaskResult(
                finished=True,
                session_id=sub.session_id,
                task_id=sub.task_id,
                reward=score.reward,
                error=outcome.error,
                termination_reason=outcome.termination_reason,
                elapsed=outcome.elapsed,
                raw_result=outcome.raw_result,
                metadata=meta,
            )

        import asyncio

        return list(await asyncio.gather(*[_run_submission(sub) for sub in submissions]))
