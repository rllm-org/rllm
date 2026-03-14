"""Terminal-Bench evaluator (stub).

Accesses the sandbox from episode.artifacts["_sandbox"] to run
verification scripts inside the container.
"""

from __future__ import annotations

import logging

from rllm.experimental.eval.types import EvalOutput, Signal
from rllm.types import Episode

logger = logging.getLogger(__name__)


class TerminalBenchEvaluator:
    """Evaluator stub for Terminal-Bench tasks.

    Full implementation will run task verification scripts inside the
    sandbox container and parse PASS/FAIL results.
    """

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        sandbox = episode.artifacts.get("_sandbox")
        if sandbox is None:
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                signals=[Signal("pass_rate", 0.0)],
                metadata={"reason": "no_sandbox"},
            )

        # TODO: Run task verification script
        # test_output = sandbox.exec("bash /tests/run-tests.sh")
        # Parse test output for PASS/FAIL

        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal("pass_rate", 0.0)],
            metadata={"reason": "stub_evaluator"},
        )
