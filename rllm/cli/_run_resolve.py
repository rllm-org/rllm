"""Resolve a :class:`RunSpec` into the objects the trainer / eval runner need.

Shared by the config-file ``rllm train`` / ``rllm eval`` drivers. Reuses the
same primitives as the flag-based paths (``load_agent`` / ``load_evaluator`` /
``DatasetRegistry`` / ``BenchmarkLoader``) so behavior matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResolvedRun:
    """Objects resolved from a :class:`RunSpec`, ready for ``AgentTrainer``."""

    agent_flow: Any = None
    evaluator: Any = None
    train_dataset: Any = None
    val_dataset: Any = None
    hooks: Any = None
    # Human-readable summaries for the CLI header.
    agent_display: str = ""
    evaluator_display: str = ""


def load_agent_with_args(name_or_path: str, agent_args: dict | None) -> Any:
    """Load an AgentFlow, applying ``agent_args``.

    Delegates to :func:`rllm.eval.agent_loader.load_agent`, which resolves the
    name/path and applies ``agent_args`` as constructor kwargs (class) or via
    the instance's ``configure()`` (already-instantiated flows).
    """
    from rllm.eval.agent_loader import load_agent

    return load_agent(name_or_path, dict(agent_args or {}))


def _resolve_dataset(name: str | None, split: str | None, *, default_split: str):
    """Load a dataset by registry name / local benchmark dir / harbor name.

    Reuses the flag-path helpers in ``rllm.cli.train`` so resolution matches.
    Returns ``(dataset, resolved_split)`` or ``(None, split)``.
    """
    if not name:
        return None, split

    from rllm.tasks.loader import BenchmarkLoader

    # Local benchmark directory (dataset.toml / task.toml).
    if BenchmarkLoader.is_local_benchmark(name):
        from rllm.data.dataset import Dataset

        bench = BenchmarkLoader.load(name)
        resolved_split = split or bench.split or default_split
        return Dataset(data=list(bench.tasks), name=bench.name, split=resolved_split), resolved_split

    # Registry / catalog / harbor name (auto-pull if needed).
    from rllm.cli._pull import load_dataset_catalog
    from rllm.cli.train import _load_or_pull_dataset, _resolve_dataset_entry

    catalog = load_dataset_catalog()
    entry = _resolve_dataset_entry(name, catalog)
    resolved_split = split
    if resolved_split is None:
        if entry and "train_split" in entry and default_split == "train":
            resolved_split = entry["train_split"]
        elif entry:
            resolved_split = entry.get("eval_split", default_split)
        else:
            resolved_split = default_split

    dataset = _load_or_pull_dataset(name, resolved_split, catalog, entry)

    # Wrap harbor rows as Tasks rooted at their task dir (per-task verifiers).
    if dataset is not None and entry and entry.get("source", "").startswith("harbor:"):
        from rllm.data.dataset import Dataset, _wrap_rows_as_tasks

        dataset = Dataset(data=_wrap_rows_as_tasks(list(dataset.data)), name=name, split=resolved_split)
    return dataset, resolved_split


def resolve_run(run, *, for_eval: bool = False) -> ResolvedRun:
    """Resolve a :class:`RunSpec` into a :class:`ResolvedRun`.

    Order of resolution mirrors the flag-based paths:

    * ``entrypoint`` (escape hatch) wins — import ``module:function``, call with
      no args, and take ``agent_flow`` / ``evaluator`` / ``train_dataset`` /
      ``val_dataset`` / ``hooks`` from its returned dict.
    * otherwise resolve ``[run.agent]`` via :func:`load_agent_with_args`,
      ``[run.agent].evaluator`` via ``load_evaluator`` (omit -> per-task
      verifier), and the datasets via :func:`_resolve_dataset`.
    """
    from rllm.config.run_config import RunSpec  # noqa: F401  (type hint only)

    resolved = ResolvedRun()

    if run.entrypoint:
        from rllm.eval.agent_loader import _import_from_path

        fn = _import_from_path(run.entrypoint)
        out = fn() if callable(fn) else fn
        if not isinstance(out, dict):
            raise TypeError(f"entrypoint {run.entrypoint!r} must return a dict of {{agent_flow, evaluator, train_dataset, val_dataset, hooks}}, got {type(out).__name__}")
        resolved.agent_flow = out.get("agent_flow")
        resolved.evaluator = out.get("evaluator")
        resolved.train_dataset = out.get("train_dataset")
        resolved.val_dataset = out.get("val_dataset")
        resolved.hooks = out.get("hooks")
        resolved.agent_display = f"entrypoint:{run.entrypoint}"
        resolved.evaluator_display = "from entrypoint" if resolved.evaluator is not None else "per-task / entrypoint"
        return resolved

    # ---- Agent ----
    if not run.agent:
        raise ValueError("config [run.agent].name (or [run].entrypoint) is required")
    resolved.agent_flow = load_agent_with_args(run.agent, run.agent_args)
    resolved.agent_display = run.agent

    # ---- Evaluator (optional; omit -> per-task verifier) ----
    if run.evaluator:
        from rllm.eval.evaluator_loader import load_evaluator

        resolved.evaluator = load_evaluator(run.evaluator)
        resolved.evaluator_display = f"{run.evaluator} (overrides per-task verifier)"
    else:
        resolved.evaluator_display = "per-task (from dataset.toml / task.toml)"

    # ---- Datasets ----
    resolved.train_dataset, _ = _resolve_dataset(run.train_dataset, run.train_split, default_split="train")

    if run.max_examples is not None and resolved.train_dataset is not None and run.max_examples < len(resolved.train_dataset):
        resolved.train_dataset = resolved.train_dataset.select(range(run.max_examples))

    if run.val_dataset:
        resolved.val_dataset, _ = _resolve_dataset(run.val_dataset, run.val_split, default_split="test")
    elif not for_eval:
        # No explicit val dataset: reuse the train tasks for validation
        # (matches the flag-based ``_run_train`` fallback).
        resolved.val_dataset = resolved.train_dataset

    return resolved
