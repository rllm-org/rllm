#!/usr/bin/env python3
"""SWE Evaluator — scores an Episode by routing to dataset-specific graders.

Implements the rllm ``Evaluator`` protocol. Receives the task dict and
the Episode produced by ``SWEAgentFlow``, grades the patch, and returns
an ``EvalOutput`` with the reward.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rllm.experimental.eval.types import EvalOutput, Signal
from rllm.types import Episode

from environment import create_env, default_scripts_dir, ensure_bootstrapped
from tasks.common import make_log

ensure_bootstrapped()

from tasks.swebench_multilingual import grade_swebench_multilingual
from tasks.swebench_pro import grade_swebench_pro
from tasks.swesmith import grade_swesmith


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
        # Grading runs in a thread pool because EvalRunner calls evaluate()
        # directly in the async event loop, but env.execute() internally uses
        # asyncio.run() which cannot be called from a running loop.
        self._grading_pool = ThreadPoolExecutor(max_workers=max_grading_workers)

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        """Score an Episode by grading its patch against the task's test suite.

        ``EvalOutput.metadata`` must only contain numeric values —
        ``AgentFlowEngine`` copies them into ``episode.metrics``.
        """
        patch = episode.artifacts["patch"]
        exit_status = episode.artifacts["exit_status"]
        env = episode.artifacts["env"]

        if exit_status != "Submitted" or not patch.strip():
            return EvalOutput(reward=0.0, is_correct=False)

        try:
            result = self._grading_pool.submit(
                self._grade, task, env, patch,
            ).result()
        except Exception as e:
            log = make_log(self.verbose)
            log(f"Grading error: {type(e).__name__}: {e}")
            return EvalOutput(reward=0.0, is_correct=False)

        if isinstance(result, dict):
            reward = result["reward"]
        else:
            reward = float(result)

        signals = []
        if isinstance(result, dict):
            for key in ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total"):
                if key in result and isinstance(result[key], (int, float)):
                    signals.append(Signal(name=key, value=float(result[key])))

        return EvalOutput(
            reward=reward,
            is_correct=reward > 0,
            signals=signals,
        )

    def _grade(self, task: dict, env: Any, patch: str) -> dict | float:
        """Run the dataset-specific grader. Executed in a thread pool."""
        eval_type = task["eval_type"]
        if eval_type == "swebench_pro":
            return grade_swebench_pro(task, patch, self.dockerhub_username, self.scripts_dir, self.verbose)
        elif eval_type == "swesmith":
            return grade_swesmith(task, patch, self._create_env, self.verbose)
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
