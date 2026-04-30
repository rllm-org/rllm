r"""Math evaluator: extract \boxed{} answer + symbolic grading.

Wraps :func:`rllm.eval.reward_fns.math.evaluate` so the cookbook owns
its own entry-point name (``rllm.evaluators.math``). The shared
evaluator pulls the model's answer from ``episode.artifacts["answer"]``
via :func:`rllm.eval.reward_fns._helpers.extract_answer_text`.
"""

from __future__ import annotations

from pathlib import Path

import rllm
from rllm.eval.reward_fns.math import evaluate as _math_evaluate
from rllm.eval.types import EvalOutput
from rllm.types import Episode, Task


@rllm.evaluator
def math_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    # The eval Runner passes a Task; the training engine passes a raw
    # row dict. The shared math evaluator reads ``task.metadata`` so
    # we wrap dicts into a synthetic Task with that dict as metadata.
    if isinstance(task, dict):
        task = Task(id="", instruction="", metadata=task, dataset_dir=Path("."))
    return _math_evaluate(task, episode)
