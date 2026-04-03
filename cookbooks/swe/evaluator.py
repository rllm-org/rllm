#!/usr/bin/env python3
"""SWE Evaluator — scores an Episode by routing to dataset-specific graders.

Implements the rllm ``Evaluator`` protocol. Receives the task dict and
the Episode produced by ``SWEAgentFlow``, grades the patch, and returns
an ``EvalOutput`` with the reward.
"""

from __future__ import annotations

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
        verbose: bool = False,
    ):
        self.scripts_dir = scripts_dir or default_scripts_dir()
        self.dockerhub_username = dockerhub_username
        self.command_timeout = command_timeout
        self.sandbox_timeout = sandbox_timeout
        self.verbose = verbose

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        """Score an Episode by grading its patch against the task's test suite.

        ``EvalOutput.metadata`` must only contain numeric values —
        ``AgentFlowEngine`` copies them into ``episode.metrics``.
        """
        log = make_log(self.verbose)
        patch = episode.artifacts["patch"]
        exit_status = episode.artifacts["exit_status"]
        eval_type = task["eval_type"]
        env = episode.artifacts["env"]

        if exit_status != "Submitted" or not patch.strip():
            return EvalOutput(reward=0.0, is_correct=False)

        try:
            if eval_type == "swebench_pro":
                result = grade_swebench_pro(
                    task, patch, self.dockerhub_username, self.scripts_dir, self.verbose,
                )
            elif eval_type == "swesmith":
                result = grade_swesmith(task, patch, self._create_env, self.verbose)
            elif eval_type == "swebench":
                result = grade_swebench_multilingual(task, env, patch, self.verbose)
            else:
                raise ValueError(f"Unsupported eval_type: {eval_type}")
        except Exception as e:
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

    def _create_env(self, task: dict):
        """Factory for fresh grading sandboxes (used by SWE-smith grader)."""
        return create_env(
            task,
            command_timeout=self.command_timeout,
            sandbox_timeout=self.sandbox_timeout,
        )
