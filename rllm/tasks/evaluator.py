"""TaskEvaluator: thin wrapper that delegates to ``Task.evaluate``.

All the actual verification logic (uploading tests, parsing reward files,
falling back to Harbor reward paths) lives on :class:`Task` itself — this
class just bridges the rLLM ``Evaluator`` protocol.
"""

from __future__ import annotations

from rllm.eval.types import EvalOutput
from rllm.tasks.task import Task
from rllm.types import Episode


class TaskEvaluator:
    """Evaluates a Task by delegating to ``Task.evaluate(sandbox)``.

    The sandbox is retrieved from ``episode.artifacts["_sandbox"]`` (set by
    ``EvalRunner`` for sandboxed agents).
    """

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        sandbox = episode.artifacts.get("_sandbox")
        task_path = task.get("task_path")
        if task_path is None:
            return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no task_path"})

        loaded = Task.load(task_path)

        if sandbox is None:
            # Fall back to pre-computed reward in artifacts (e.g., Harbor integration)
            if "harbor_reward" in episode.artifacts:
                reward = float(episode.artifacts["harbor_reward"])
                is_correct = bool(episode.artifacts.get("harbor_is_correct", reward >= 1.0))
                return EvalOutput(reward=reward, is_correct=is_correct)
            return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no sandbox available"})

        return loaded.evaluate(sandbox)
