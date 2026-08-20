"""Per-file Episode storage for eval runs.

When ``rllm eval`` runs with ``--save-episodes`` (the default), each
:class:`~rllm.types.Episode` is written to its own JSON file under::

    <run_dir>/
        meta.json                  # dataset, model, agent, timestamp
        episodes/
            episode_000000_<task_id>.json
            episode_000001_<task_id>.json
            ...

The aggregate :class:`~rllm.eval.results.EvalResult` JSON sits next to
``run_dir`` (same parent directory, same ``<dataset>_<model>_<timestamp>``
basename) so the two are paired by name.

The store is consumed by :mod:`rllm.eval.visualizer` for read-back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rllm.eval.results import EvalItem
from rllm.types import INFRA_ERROR_REASONS, Episode
from rllm.workflows.workflow import TerminationReason


def _sanitize(s: str) -> str:
    """Make ``s`` safe to use as a single path component."""
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in "-_.":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "_"


def _json_default(obj: Any) -> Any:
    """Fallback encoder for objects Pydantic's json mode leaves opaque.

    Mirrors :meth:`rllm.utils.tracking.UILogger._json_serializer` for numpy
    types but additionally expands dataclasses (notably :class:`Task`) into
    dicts so the visualizer can render structured fields like
    ``task.instruction`` rather than a single ``repr`` string.
    """
    import dataclasses

    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


class EvalEpisodeStore:
    """Writes one JSON file per :class:`Episode` under ``<run_dir>/episodes/``."""

    META_FILENAME = "meta.json"
    EPISODES_SUBDIR = "episodes"
    PROGRESS_SUBDIR = "progress"

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # ``episodes/`` is created lazily on the first ``write()`` call so
        # callers that only need ``write_meta()`` (e.g. ``--no-save-episodes``)
        # don't leave an empty subdirectory behind.
        self.episodes_dir = self.run_dir / self.EPISODES_SUBDIR
        self.progress_dir = self.run_dir / self.PROGRESS_SUBDIR

    def write_meta(self, meta: dict[str, Any]) -> Path:
        """Write run-level metadata to ``<run_dir>/meta.json``."""
        path = self.run_dir / self.META_FILENAME
        temporary = path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=_json_default)
            f.write("\n")
        temporary.replace(path)
        return path

    def read_meta(self) -> dict[str, Any]:
        path = self.run_dir / self.META_FILENAME
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else {}

    def episode_path(self, idx: int, episode: Episode) -> Path:
        """Compute the file path for ``episode`` at index ``idx``."""
        task = episode.task
        task_id = getattr(task, "id", None) if task is not None else None
        if task_id is None:
            task_id = episode.id
        return self.episodes_dir / f"episode_{idx:06d}_{_sanitize(task_id)}.json"

    def write(self, idx: int, episode: Episode) -> Path:
        """Serialize ``episode`` to its own JSON file and return the path."""
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        path = self.episode_path(idx, episode)
        data = episode.model_dump(mode="json")
        # Stamp the eval-time idx into the saved episode so consumers
        # (e.g. the visualizer) can cross-reference EvalResult.items.
        data["eval_idx"] = idx
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)
        return path

    @staticmethod
    def item_from_episode(idx: int, attempt: int, episode: Episode, *, task_id: str | None = None) -> EvalItem:
        error = None
        # Infra failures (model/proxy/sandbox/grading breakdowns) are not real
        # attempts: mark them as errors so a resumed run re-executes them instead
        # of treating the empty rollout as a completed result.
        if episode.termination_reason in INFRA_ERROR_REASONS or episode.termination_reason == TerminationReason.TIMEOUT:
            raw_error = (episode.metadata or {}).get("error") or {}
            error = raw_error.get("message") if isinstance(raw_error, dict) else str(raw_error)
            error = error or "episode terminated with an error"
        trajectory = episode.trajectories[0] if episode.trajectories else None
        reward = float(trajectory.reward or 0.0) if trajectory is not None else 0.0
        signals = dict(trajectory.signals or {}) if trajectory is not None else {}
        task = episode.task
        if task_id is None:
            task_id = getattr(task, "id", None)
            if task_id is None and isinstance(task, dict):
                task_id = task.get("id") or task.get("TASK") or task.get("task_id")
        return EvalItem(
            idx=idx,
            attempt=attempt,
            task_id=str(task_id) if task_id is not None else None,
            reward=reward,
            is_correct=bool(episode.is_correct),
            error=error,
            signals=signals,
        )

    def write_progress(self, flat_idx: int, idx: int, attempt: int, episode: Episode, *, task_id: str | None = None) -> Path:
        """Atomically persist a compact completed-rollout record."""
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        item = self.item_from_episode(idx, attempt, episode, task_id=task_id)
        path = self.progress_dir / f"item_{flat_idx:06d}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(item.__dict__, indent=2, default=_json_default) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    def load_completed_items(self, *, successful_only: bool = True) -> list[EvalItem]:
        items: list[EvalItem] = []
        if not self.progress_dir.is_dir():
            return items
        seen: set[tuple[int, int]] = set()
        for path in sorted(self.progress_dir.glob("item_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                item = EvalItem(**data)
            except (OSError, ValueError, TypeError):
                continue
            key = (item.idx, item.attempt)
            if key in seen or (successful_only and item.error is not None):
                continue
            seen.add(key)
            items.append(item)
        return items
