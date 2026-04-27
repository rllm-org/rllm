"""Evaluator loader: resolves evaluator by registry name, import path, or entry point.

The built-in registry maps reward-fn names to score_fn module import paths.
A bare function under that path is wrapped as an Evaluator at load time so
the existing ``Evaluator`` protocol callers (``EvalRunner``, etc.) keep
working without changes.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from importlib.metadata import entry_points
from typing import Any

from rllm.eval.types import EvalOutput, Evaluator, Signal

_RLLM_HOME = os.environ.get("RLLM_HOME", os.path.expanduser("~/.rllm"))
_USER_EVALUATORS_FILE = os.path.join(_RLLM_HOME, "evaluators.json")


def _load_user_evaluators() -> dict[str, dict]:
    """Load the user-registered evaluators from ~/.rllm/evaluators.json."""
    if not os.path.exists(_USER_EVALUATORS_FILE):
        return {}
    try:
        with open(_USER_EVALUATORS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_user_evaluators(registry: dict[str, dict]) -> None:
    """Persist the user-registered evaluators to ~/.rllm/evaluators.json."""
    os.makedirs(os.path.dirname(_USER_EVALUATORS_FILE), exist_ok=True)
    with open(_USER_EVALUATORS_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_evaluator(name: str, evaluator_or_path: Evaluator | type | str) -> None:
    """Persist an evaluator registration so it's discoverable across processes.

    Args:
        name: Short name for the evaluator (e.g. ``"relevance"``).
        evaluator_or_path: One of:
            - An import-path string (e.g. ``"my_eval:RelevanceEvaluator"``).
            - A class — the import path is derived automatically.
            - An instance — the import path is derived from its class.
    """
    if isinstance(evaluator_or_path, str):
        import_path = evaluator_or_path
    elif isinstance(evaluator_or_path, type):
        _validate_evaluator_class(evaluator_or_path, name)
        import_path = f"{evaluator_or_path.__module__}:{evaluator_or_path.__qualname__}"
    else:
        _validate_evaluator(evaluator_or_path, name)
        cls = type(evaluator_or_path)
        import_path = f"{cls.__module__}:{cls.__qualname__}"

    registry = _load_user_evaluators()
    registry[name] = {"import_path": import_path}
    _save_user_evaluators(registry)


def unregister_evaluator(name: str) -> bool:
    registry = _load_user_evaluators()
    if name not in registry:
        return False
    del registry[name]
    _save_user_evaluators(registry)
    return True


def _validate_evaluator(obj: object, name: str) -> None:
    if not hasattr(obj, "evaluate") or not callable(obj.evaluate):
        raise TypeError(f"Evaluator '{name}' must have an .evaluate() method, got {type(obj).__name__}")


def _validate_evaluator_class(cls: type, name: str) -> None:
    eval_attr = getattr(cls, "evaluate", None)
    if eval_attr is None or not callable(eval_attr):
        raise TypeError(f"Evaluator '{name}' must be a class with an .evaluate() method, got {cls.__name__}")


# ---------------------------------------------------------------------------
# Built-in registry: name → import path of a score_fn evaluate() function
# ---------------------------------------------------------------------------

_EVALUATOR_REGISTRY: dict[str, str] = {
    "math_reward_fn": "rllm.eval.reward_fns.math:evaluate",
    "countdown_reward_fn": "rllm.eval.reward_fns.countdown:evaluate",
    "code_reward_fn": "rllm.eval.reward_fns.code:evaluate",
    "f1_reward_fn": "rllm.eval.reward_fns.f1:evaluate",
    "mcq_reward_fn": "rllm.eval.reward_fns.mcq:evaluate",
    "ifeval_reward_fn": "rllm.eval.reward_fns.ifeval:evaluate",
    "bfcl_reward_fn": "rllm.eval.reward_fns.bfcl:evaluate",
    "llm_judge_reward_fn": "rllm.eval.reward_fns.llm_judge:evaluate",
    "llm_equality_reward_fn": "rllm.eval.reward_fns.llm_equality:evaluate",
    "translation_reward_fn": "rllm.eval.reward_fns.translation:evaluate",
    "widesearch_reward_fn": "rllm.eval.reward_fns.widesearch:evaluate",
    "iou_reward_fn": "rllm.eval.reward_fns.iou:evaluate",
    "point_in_mask_reward_fn": "rllm.eval.reward_fns.point_in_mask:evaluate",
    "depth_reward_fn": "rllm.eval.reward_fns.depth:evaluate",
}

# Lazy-loaded evaluators (optional deps)
_LAZY_EVALUATOR_REGISTRY: dict[str, str] = {
    "harbor_reward_fn": "rllm.integrations.harbor.evaluator:HarborEvaluator",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_dataset_catalog() -> dict:
    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "registry",
        "datasets.json",
    )
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def _load_and_instantiate(import_path: str, name: str) -> Evaluator:
    """Resolve ``module:attr`` to an Evaluator instance.

    - Class → instantiate.
    - Bare function → wrap as ``_FunctionEvaluator`` so it satisfies the
      ``Evaluator`` protocol (callable with ``.evaluate(task, episode)``).
    - Instance with ``.evaluate()`` → use directly.
    """
    module_path, attr_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    obj: Any = getattr(module, attr_name)

    if isinstance(obj, type):
        obj = obj()
    elif callable(obj) and not hasattr(obj, "evaluate"):
        # Bare function — wrap so it has .evaluate()
        obj = _FunctionEvaluator(obj)

    _validate_evaluator(obj, name)
    return obj


class _FunctionEvaluator:
    """Wrap a score_fn ``evaluate(task, episode)`` callable as an Evaluator.

    Score_fns expect ``task`` to be an ``rllm.task.Task`` object. Legacy
    callers (``EvalRunner``) pass a dict. We auto-wrap the dict in a Task
    so both call styles work.
    """

    def __init__(self, fn):
        self.fn = fn
        sig = inspect.signature(fn)
        self._params = list(sig.parameters.values())

    def evaluate(self, task, episode):
        # Legacy callers pass a dict; reward_fns expect Task. Adapt on the fly.
        if isinstance(task, dict):
            from pathlib import Path

            from rllm.task import Task

            task = Task(id="", instruction="", metadata=task, benchmark_dir=Path("/"))

        # Some reward_fns may use kwarg names like 'metadata'/'trajectory'
        # rather than 'task'/'episode'. Inspect and dispatch accordingly.
        kwargs = {}
        for p in self._params:
            if p.name == "task":
                kwargs["task"] = task
            elif p.name == "metadata":
                kwargs["metadata"] = task.metadata
            elif p.name == "episode":
                kwargs["episode"] = episode
            elif p.name == "trajectory":
                kwargs["trajectory"] = _trajectory_as_dict(episode)

        if not kwargs:
            kwargs = {"task": task, "episode": episode}

        result = self.fn(**kwargs)
        return _coerce_eval_result(result)


def _trajectory_as_dict(episode) -> dict:
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


def _coerce_eval_result(result) -> EvalOutput:
    if isinstance(result, EvalOutput):
        return result
    if isinstance(result, bool):
        return EvalOutput(reward=1.0 if result else 0.0, is_correct=result)
    if isinstance(result, int | float):
        r = float(result)
        return EvalOutput(reward=r, is_correct=r >= 1.0)
    if isinstance(result, tuple) and len(result) == 2:
        r, ok = result
        return EvalOutput(reward=float(r), is_correct=bool(ok))
    if isinstance(result, dict):
        r = float(result.get("reward", 0.0))
        ok = result.get("is_correct", r >= 1.0)
        signals = [Signal(name=k, value=float(v)) for k, v in result.get("signals", {}).items()]
        return EvalOutput(reward=r, is_correct=ok, signals=signals, metadata=result.get("metadata", {}))
    try:
        r = float(result)
        return EvalOutput(reward=r, is_correct=r >= 1.0)
    except (TypeError, ValueError):
        return EvalOutput(reward=0.0, is_correct=False, metadata={"error": f"cannot coerce {type(result).__name__}"})


def load_evaluator(name_or_path: str) -> Evaluator:
    """Load an evaluator by registry name, import path, or entry point."""
    # 1. User-registered evaluators (persistent)
    user_evaluators = _load_user_evaluators()
    if name_or_path in user_evaluators:
        return _load_and_instantiate(user_evaluators[name_or_path]["import_path"], name_or_path)

    # 2. Explicit import path
    if ":" in name_or_path:
        return _load_and_instantiate(name_or_path, name_or_path)

    # 3. Built-in registry (name → score_fn import path)
    if name_or_path in _EVALUATOR_REGISTRY:
        return _load_and_instantiate(_EVALUATOR_REGISTRY[name_or_path], name_or_path)

    # 3b. Lazy-loaded evaluators (optional deps)
    if name_or_path in _LAZY_EVALUATOR_REGISTRY:
        return _load_and_instantiate(_LAZY_EVALUATOR_REGISTRY[name_or_path], name_or_path)

    # 4. Plugin discovery via entry points
    eps = entry_points(group="rllm.evaluators")
    for ep in eps:
        if ep.name == name_or_path:
            obj = ep.load()
            if isinstance(obj, type):
                obj = obj()
            elif callable(obj) and not hasattr(obj, "evaluate"):
                obj = _FunctionEvaluator(obj)
            _validate_evaluator(obj, name_or_path)
            return obj

    available = ", ".join(sorted(_EVALUATOR_REGISTRY.keys()))
    raise KeyError(f"Evaluator '{name_or_path}' not found. Available built-in: {available}")


def resolve_evaluator_from_catalog(benchmark: str) -> Evaluator | None:
    """Auto-resolve an evaluator from the datasets.json reward_fn field."""
    try:
        catalog = _load_dataset_catalog()
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    entry = catalog.get("datasets", {}).get(benchmark)
    if entry is None:
        return None

    reward_fn_name = entry.get("reward_fn")
    if reward_fn_name is None:
        return None

    try:
        return load_evaluator(reward_fn_name)
    except KeyError:
        return None


def list_evaluators() -> list[dict]:
    results: list[dict] = []

    for name, info in sorted(_load_user_evaluators().items()):
        results.append({"name": name, "source": "registered", "type": info["import_path"]})

    seen = {r["name"] for r in results}
    for name, import_path in sorted(_EVALUATOR_REGISTRY.items()):
        if name not in seen:
            results.append({"name": name, "source": "built-in", "type": import_path})

    seen = {r["name"] for r in results}
    eps = entry_points(group="rllm.evaluators")
    for ep in eps:
        if ep.name not in seen:
            pkg = ep.dist.name if ep.dist else "unknown"
            results.append({"name": ep.name, "source": f"plugin ({pkg})", "type": str(ep.value)})

    return results
