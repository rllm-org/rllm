"""Evaluator loader: resolves evaluator by registry name, import path, or entry point."""

from __future__ import annotations

import importlib
import json
import os
from importlib.metadata import entry_points

from rllm.experimental.eval.bfcl_evaluator import BFCLEvaluator
from rllm.experimental.eval.ifeval_evaluator import IFEvalEvaluator
from rllm.experimental.eval.llm_equality_evaluator import LLMEqualityEvaluator
from rllm.experimental.eval.llm_judge_evaluator import LLMJudgeEvaluator
from rllm.experimental.eval.translation_evaluator import TranslationEvaluator
from rllm.experimental.eval.types import (
    CodeEvaluator,
    CountdownEvaluator,
    DepthEvaluator,
    Evaluator,
    F1Evaluator,
    IoUEvaluator,
    MathEvaluator,
    MCQEvaluator,
    PointInMaskEvaluator,
)
from rllm.experimental.eval.widesearch_evaluator import WideSearchEvaluator

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
    """Persist an evaluator registration so it's discoverable by name across processes.

    The registration is saved to ``~/.rllm/evaluators.json``.  After calling
    this, ``load_evaluator(name)`` and ``rllm eval --evaluator <name>`` will
    resolve the evaluator from a new Python process or CLI session.

    Args:
        name: Short name for the evaluator (e.g., ``"relevance"``).
        evaluator_or_path: One of:
            - An import-path string (e.g., ``"my_eval:RelevanceEvaluator"``).
            - A class — the import path is derived automatically.
            - An instance — the import path is derived from its class.

    Raises:
        TypeError: If an instance/class is provided but doesn't have ``.evaluate()``.
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
    """Remove a user-registered evaluator.

    Returns:
        True if the evaluator was found and removed, False otherwise.
    """
    registry = _load_user_evaluators()
    if name not in registry:
        return False
    del registry[name]
    _save_user_evaluators(registry)
    return True


def _validate_evaluator(obj: object, name: str) -> None:
    """Validate that an object conforms to the Evaluator protocol."""
    if not hasattr(obj, "evaluate") or not callable(obj.evaluate):
        raise TypeError(f"Evaluator '{name}' must have an .evaluate() method, got {type(obj).__name__}")


def _validate_evaluator_class(cls: type, name: str) -> None:
    """Validate that a class has an .evaluate method."""
    eval_attr = getattr(cls, "evaluate", None)
    if eval_attr is None or not callable(eval_attr):
        raise TypeError(f"Evaluator '{name}' must be a class with an .evaluate() method, got {cls.__name__}")


_EVALUATOR_REGISTRY: dict[str, type] = {
    "math_reward_fn": MathEvaluator,
    "countdown_reward_fn": CountdownEvaluator,
    "code_reward_fn": CodeEvaluator,
    "f1_reward_fn": F1Evaluator,
    "mcq_reward_fn": MCQEvaluator,
    "ifeval_reward_fn": IFEvalEvaluator,
    "bfcl_reward_fn": BFCLEvaluator,
    "llm_judge_reward_fn": LLMJudgeEvaluator,
    "llm_equality_reward_fn": LLMEqualityEvaluator,
    "translation_reward_fn": TranslationEvaluator,
    "widesearch_reward_fn": WideSearchEvaluator,
    "iou_reward_fn": IoUEvaluator,
    "point_in_mask_reward_fn": PointInMaskEvaluator,
    "depth_reward_fn": DepthEvaluator,
}


def _load_dataset_catalog() -> dict:
    """Load the datasets.json catalog from the registry directory."""
    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "registry",
        "datasets.json",
    )
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def _load_and_instantiate(import_path: str, name: str) -> Evaluator:
    """Import an evaluator from a path, auto-instantiate classes, and validate."""
    module_path, attr_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, attr_name)
    if isinstance(obj, type):
        obj = obj()
    _validate_evaluator(obj, name)
    return obj


def load_evaluator(name_or_path: str) -> Evaluator:
    """Load an evaluator by registry name, import path, or entry point.

    Lookup order: user registry (``~/.rllm/evaluators.json``) → colon import
    path → built-in registry → entry points (``rllm.evaluators`` group).

    Args:
        name_or_path: A registry name (e.g., ``"math_reward_fn"``), a
            colon-separated import path (e.g., ``"my_module:MyEvaluator"``),
            or a plugin name registered via the ``rllm.evaluators``
            entry-point group.

    Returns:
        An Evaluator instance with an ``.evaluate()`` method.
    """
    # 1. User-registered evaluators (persistent, from register_evaluator())
    user_evaluators = _load_user_evaluators()
    if name_or_path in user_evaluators:
        return _load_and_instantiate(user_evaluators[name_or_path]["import_path"], name_or_path)

    # 2. Explicit import path: "my_module:MyEvaluator"
    if ":" in name_or_path:
        return _load_and_instantiate(name_or_path, name_or_path)

    # 3. Built-in registry
    if name_or_path in _EVALUATOR_REGISTRY:
        return _EVALUATOR_REGISTRY[name_or_path]()

    # 4. Plugin discovery via entry points
    eps = entry_points(group="rllm.evaluators")
    for ep in eps:
        if ep.name == name_or_path:
            obj = ep.load()
            if isinstance(obj, type):
                obj = obj()
            _validate_evaluator(obj, name_or_path)
            return obj

    available = ", ".join(sorted(_EVALUATOR_REGISTRY.keys()))
    raise KeyError(f"Evaluator '{name_or_path}' not found in registry or agenthub. Available built-in: {available}")


def resolve_evaluator_from_catalog(benchmark: str) -> Evaluator | None:
    """Auto-resolve an evaluator from the datasets.json reward_fn field.

    Args:
        benchmark: Dataset name (e.g., "gsm8k").

    Returns:
        An Evaluator instance if the benchmark's reward_fn maps to a known evaluator,
        None otherwise.
    """
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

    evaluator_cls = _EVALUATOR_REGISTRY.get(reward_fn_name)
    if evaluator_cls is not None:
        return evaluator_cls()

    # Fall through to full loader (user registry, import paths, entry points)
    try:
        return load_evaluator(reward_fn_name)
    except KeyError:
        return None


def list_evaluators() -> list[dict]:
    """List all available evaluators (user-registered + built-in + agenthub).

    Returns:
        A list of dicts with keys: name, source, type.
    """
    results: list[dict] = []

    # User-registered evaluators
    for name, info in sorted(_load_user_evaluators().items()):
        results.append(
            {
                "name": name,
                "source": "registered",
                "type": info["import_path"],
            }
        )

    # Built-in evaluators from registry
    seen_names = {r["name"] for r in results}
    for name, cls in sorted(_EVALUATOR_REGISTRY.items()):
        if name not in seen_names:
            results.append(
                {
                    "name": name,
                    "source": "built-in",
                    "type": f"{cls.__module__}.{cls.__name__}",
                }
            )

    # Plugin evaluators from entry points
    seen_names = {r["name"] for r in results}
    eps = entry_points(group="rllm.evaluators")
    for ep in eps:
        if ep.name not in seen_names:
            pkg = ep.dist.name if ep.dist else "unknown"
            results.append(
                {
                    "name": ep.name,
                    "source": f"plugin ({pkg})",
                    "type": str(ep.value),
                }
            )

    return results
