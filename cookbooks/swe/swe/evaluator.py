#!/usr/bin/env python3
"""SWE Evaluator — scores an Episode by routing to dataset-specific graders.

Implements the rllm ``Evaluator`` protocol. Receives the task dict and
the Episode produced by ``SWEAgentFlow``, grades the patch, and returns
an ``EvalOutput`` with the reward.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode

from swe.environment import create_env, default_scripts_dir, ensure_bootstrapped
from swe.tasks.common import make_log
from swe.utils import close_env

ensure_bootstrapped()

from swe.tasks.swebench_multilingual import grade_swebench_multilingual
from swe.tasks.swebench_pro import grade_swebench_pro
from swe.tasks.swesmith import grade_swesmith
from swe.tasks.swe_rebench_v2 import grade_swe_rebench_v2


class SWEEvaluator:
    """Scores SWE Episodes by routing to the correct dataset grader."""

    def __init__(
        self,
        scripts_dir: str | None = None,
        dockerhub_username: str = "jefzda",
        command_timeout: int = 120,
        sandbox_timeout: int = 3600,
        max_grading_workers: int = 256,
        verbose: bool = False,
    ):
        self.scripts_dir = scripts_dir or default_scripts_dir()
        self.dockerhub_username = dockerhub_username
        self.command_timeout = command_timeout
        self.sandbox_timeout = sandbox_timeout
        self.verbose = verbose
        self._max_grading_workers = max_grading_workers
        # Grading runs in a thread pool because EvalRunner calls evaluate()
        # directly in the async event loop, but env.execute() internally uses
        # asyncio.run() which cannot be called from a running loop.
        # Created lazily to allow Ray serialization (ThreadPoolExecutor is not picklable).
        self._grading_pool: ThreadPoolExecutor | None = None

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        """Score an Episode by grading its patch against the task's test suite.

        ``EvalOutput.metadata`` must only contain numeric values —
        ``AgentFlowEngine`` copies them into ``episode.metrics``.

        Always closes the sandbox env after grading to prevent leaks.
        """
        patch = episode.artifacts["patch"]
        exit_status = episode.artifacts["exit_status"]
        env = episode.artifacts.get("env")

        try:
            if exit_status != "Submitted" or not patch.strip():
                return EvalOutput(reward=0.0, is_correct=False)

            if self._grading_pool is None:
                self._grading_pool = ThreadPoolExecutor(max_workers=self._max_grading_workers)

            try:
                result = self._grading_pool.submit(self._grade, task, env, patch).result()
            except Exception as e:
                make_log(self.verbose)(f"Grading error: {type(e).__name__}: {e}")
                return EvalOutput(reward=0.0, is_correct=False)

            signals = [
                Signal(name=key, value=float(result[key]))
                for key in ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total")
                if key in result
            ]
            return EvalOutput(
                reward=result["reward"],
                is_correct=result["reward"] > 0,
                signals=signals,
            )
        finally:
            if env is not None:
                close_env(env)
                episode.artifacts["env"] = None

    def _grade(self, task: dict, env, patch: str) -> dict:
        """Run the dataset-specific grader. Executed in a thread pool."""
        eval_type = task["eval_type"]
        if eval_type == "swebench_pro":
            return grade_swebench_pro(
                task, patch, self.dockerhub_username, self.scripts_dir, self.verbose,
            )
        elif eval_type == "swesmith":
            return grade_swesmith(task, patch, self._create_env, self.verbose)
        elif eval_type == "swe_rebench_v2":
            return grade_swe_rebench_v2(task, env, patch, self.verbose)
        elif eval_type == "swebench":
            return grade_swebench_multilingual(task, env, patch, self.verbose)
        else:
            raise ValueError(f"Unsupported eval_type: {eval_type}")

    def _create_env(self, task: dict):
        """Factory for fresh grading sandboxes (used by SWE-smith grader)."""
        return create_env(
            task,
            command_timeout=self.command_timeout,
            sandbox_timeout=self.sandbox_timeout,
        )
