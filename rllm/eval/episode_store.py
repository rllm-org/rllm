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

The store is consumed by the rLLM Console (:mod:`rllm.console`) for read-back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rllm.types import Episode


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

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # ``episodes/`` is created lazily on the first ``write()`` call so
        # callers that only need ``write_meta()`` (e.g. ``--no-save-episodes``)
        # don't leave an empty subdirectory behind.
        self.episodes_dir = self.run_dir / self.EPISODES_SUBDIR

    def write_meta(self, meta: dict[str, Any]) -> Path:
        """Write run-level metadata to ``<run_dir>/meta.json``."""
        path = self.run_dir / self.META_FILENAME
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=_json_default)
        return path

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
