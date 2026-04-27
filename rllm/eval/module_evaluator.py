"""PythonModuleEvaluator: load a Python verifier from the benchmark directory.

Used when ``dataset.toml`` declares ``[verifier].module = "tests.evaluate"``
or when ``tests/evaluate.py`` is auto-detected. Imports the file, finds the
``evaluate`` function (or one named in ``[verifier].function``), and wraps
it as an :class:`~rllm.eval.types.Evaluator`.

Supports two function signatures:

  - ``evaluate(task: Task, episode: Episode) -> EvalOutput``  (preferred)
  - ``evaluate(metadata: dict, trajectory: dict) -> dict``     (lightweight)

The wrapper inspects the function and adapts at call time so user code
can use whichever shape is convenient.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from collections.abc import Callable
from pathlib import Path

from rllm.eval.types import EvalOutput, Signal
from rllm.task import Task
from rllm.types import Episode

logger = logging.getLogger(__name__)


class PythonModuleEvaluator:
    """Wrap a Python ``evaluate()`` function loaded from a benchmark dir."""

    def __init__(self, fn: Callable, module_name: str = "<unknown>"):
        self.fn = fn
        self.module_name = module_name
        # Inspect signature once
        sig = inspect.signature(fn)
        self._params = list(sig.parameters.values())

    @classmethod
    def from_module(
        cls,
        benchmark_dir: Path,
        module_path: str = "tests.evaluate",
        function: str = "evaluate",
    ) -> PythonModuleEvaluator:
        """Load ``benchmark_dir/<module_path>.py`` and grab ``function``.

        ``module_path`` may be either dotted (``"tests.evaluate"``) or a
        path-like (``"tests/evaluate.py"``).
        """
        # Normalise to file path
        if module_path.endswith(".py"):
            file_path = benchmark_dir / module_path
        else:
            file_path = benchmark_dir / (module_path.replace(".", "/") + ".py")

        if not file_path.exists():
            raise FileNotFoundError(f"Verifier module not found: {file_path}")

        spec = importlib.util.spec_from_file_location(f"_rllm_verifier_{benchmark_dir.name}", str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load verifier from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fn = getattr(module, function, None)
        if fn is None:
            raise AttributeError(f"Function '{function}' not found in {file_path}")
        return cls(fn, module_name=str(file_path))

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        kwargs = self._build_kwargs(task, episode)
        try:
            result = self.fn(**kwargs)
        except Exception as e:
            logger.exception("Verifier %s raised: %s", self.module_name, e)
            return EvalOutput(reward=0.0, is_correct=False, metadata={"error": f"verifier exception: {e}"})

        return _coerce_eval_result(result)

    # ------------------------------------------------------------------

    def _build_kwargs(self, task: Task, episode: Episode) -> dict:
        """Map (task, episode) onto whatever signature the user wrote.

        Recognised parameter names (in priority order):
          - ``task`` → the Task object
          - ``metadata`` → ``task.metadata`` dict
          - ``episode`` → the Episode object
          - ``trajectory`` → ``episode.trajectories[-1]`` as a plain dict
        Any remaining positional params get filled in the natural order.
        """
        out: dict = {}
        for p in self._params:
            name = p.name
            if name == "task":
                out[name] = task
            elif name == "metadata":
                out[name] = task.metadata
            elif name == "episode":
                out[name] = episode
            elif name == "trajectory":
                out[name] = _trajectory_as_dict(episode)
        # If nothing matched, fall back to (task, episode) positionally
        if not out:
            return {"task": task, "episode": episode}
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trajectory_as_dict(episode: Episode) -> dict:
    """Serialize the latest trajectory into a plain dict for verifier scripts.

    Uses the convention that the agent's final answer is in ``trajectory.output``.
    """
    if not episode.trajectories:
        return {"output": "", "steps": []}
    traj = episode.trajectories[-1]
    return {
        "uid": traj.uid,
        "name": traj.name,
        "task": traj.task,
        "output": traj.output or "",
        "steps": [{"id": s.id, "input": s.input, "output": s.output, "reward": s.reward} for s in traj.steps],
    }


def _coerce_eval_result(result: object) -> EvalOutput:
    """Coerce common return types into an EvalOutput."""
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

    try:
        reward = float(result)  # type: ignore[arg-type]
        return EvalOutput(reward=reward, is_correct=reward >= 1.0)
    except (TypeError, ValueError):
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            metadata={"error": f"Cannot coerce {type(result).__name__} to reward"},
        )
