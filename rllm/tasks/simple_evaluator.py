"""SimpleEvaluator: wraps a user-provided Python evaluate function.

Used for "simple" datasets (gsm8k-style) where evaluation is a pure
Python function ``(task_dict, answer_str) -> reward``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

from rllm.eval.types import EvalOutput, Signal, _extract_agent_answer
from rllm.types import Episode


class SimpleEvaluator:
    """Evaluator that delegates to a user-provided Python function.

    The function signature should be ``(task: dict, answer: str) -> result``
    where *result* can be:

    - ``float`` — used as reward, ``is_correct = reward >= 1.0``
    - ``bool`` — ``True`` → reward 1.0, ``False`` → reward 0.0
    - ``dict`` — must contain ``"reward"``; may contain ``"is_correct"``, ``"signals"``
    - ``EvalOutput`` — passed through directly
    """

    def __init__(self, fn: Callable):
        self.fn = fn

    @classmethod
    def from_file(cls, path: str | Path, function_name: str = "evaluate") -> SimpleEvaluator:
        """Import a function from a standalone Python file.

        Args:
            path: Absolute or relative path to the ``.py`` file.
            function_name: Name of the function to load from the module.
        """
        path = Path(path).resolve()
        spec = importlib.util.spec_from_file_location("_user_eval", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, function_name, None)
        if fn is None:
            raise AttributeError(f"Function '{function_name}' not found in {path}")
        return cls(fn)

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        answer = _extract_agent_answer(episode)
        result = self.fn(task, answer)
        return _coerce_eval_result(result)


def _coerce_eval_result(result: object) -> EvalOutput:
    """Coerce various return types into an EvalOutput."""
    if isinstance(result, EvalOutput):
        return result

    if isinstance(result, bool):
        return EvalOutput(reward=1.0 if result else 0.0, is_correct=result)

    if isinstance(result, int | float):
        reward = float(result)
        return EvalOutput(reward=reward, is_correct=reward >= 1.0)

    if isinstance(result, tuple) and len(result) == 2:
        reward, is_correct = result
        return EvalOutput(reward=float(reward), is_correct=bool(is_correct))

    if isinstance(result, dict):
        reward = float(result.get("reward", 0.0))
        is_correct = result.get("is_correct", reward >= 1.0)
        signals = [Signal(name=k, value=float(v)) for k, v in result.get("signals", {}).items()]
        return EvalOutput(
            reward=reward,
            is_correct=is_correct,
            signals=signals,
            metadata=result.get("metadata", {}),
        )

    # Last resort: try to float-ify
    try:
        reward = float(result)  # type: ignore[arg-type]
        return EvalOutput(reward=reward, is_correct=reward >= 1.0)
    except (TypeError, ValueError):
        return EvalOutput(reward=0.0, is_correct=False, metadata={"error": f"Cannot coerce {type(result).__name__} to reward"})
