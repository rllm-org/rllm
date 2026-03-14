"""SWE-bench evaluator: wraps the swebench harness for patch grading."""

from __future__ import annotations

import logging

from rllm.experimental.eval.types import EvalOutput, Signal
from rllm.types import Episode

logger = logging.getLogger(__name__)


class SWEBenchEvaluator:
    """Evaluator that grades SWE-bench patches using the swebench harness."""

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        from rllm.rewards.code_utils.swebench import swebench_check_correctness

        patch = episode.artifacts.get("patch", "")
        instance_id = task.get("instance_id", "")

        if not patch:
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                signals=[Signal("resolve_rate", 0.0)],
                metadata={"reason": "empty_patch"},
            )

        # Build metadata expected by swebench_check_correctness
        metadata = {
            "tests": {
                "instance_id": instance_id,
            },
        }

        try:
            # swebench_check_correctness expects the patch as a raw git diff
            # in model_response — it extracts from "diff --git" onwards
            resolve_rate = swebench_check_correctness(
                model_response=patch,
                metadata=metadata,
            )
        except Exception:
            logger.exception("swebench_check_correctness failed for %s", instance_id)
            resolve_rate = 0.0

        return EvalOutput(
            reward=resolve_rate,
            is_correct=resolve_rate >= 1.0,
            signals=[Signal("resolve_rate", resolve_rate)],
        )
