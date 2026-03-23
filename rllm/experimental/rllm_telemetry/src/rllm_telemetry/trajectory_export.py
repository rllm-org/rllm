"""Export native rllm Trajectory/Episode objects to NDJSON files.

Bridges the main rllm project's trajectory objects with the rllm_telemetry
NDJSON wire format.  Uses duck typing so that ``rllm`` is **not** a
runtime dependency of ``rllm_telemetry``.

Example::

    from rllm_telemetry import export_trajectories

    # After the agentic loop finishes:
    path = export_trajectories(episode)
    # -> Trajectory file saved: /abs/path/to/trajectories/20260307_143022_a1b2c3d4.ndjson

    # With a custom output directory:
    path = export_trajectories(trajectories, output_dir="/tmp/runs")
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

from .schemas import SpanType

_DEFAULT_OUTPUT_DIR = "./trajectories"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _safe_serialize(obj: Any) -> Any:
    """Best-effort conversion of *obj* to a JSON-safe value."""
    if obj is None:
        return None
    if isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, list | tuple):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


# ---------------------------------------------------------------------------
# Per-object converters (duck-typed)
# ---------------------------------------------------------------------------


def _extract_step(step: Any, trajectory_uid: str, step_idx: int) -> dict[str, Any]:
    """Convert a single Step into an envelope-data dict."""
    data: dict[str, Any] = {
        "trajectory_uid": trajectory_uid,
        "step_idx": step_idx,
    }

    # Base Step fields (rllm.types.Step)
    for field in ("id", "input", "output", "action", "reward", "done", "metadata"):
        val = getattr(step, field, None)
        if val is not None:
            data[field] = _safe_serialize(val)

    # Training Step fields (rllm.agents.agent.Step)
    for field in (
        "observation",
        "thought",
        "model_response",
        "chat_completions",
        "mc_return",
    ):
        val = getattr(step, field, None)
        if val is not None and val != "" and val != []:
            data[field] = _safe_serialize(val)

    advantage = getattr(step, "advantage", None)
    if advantage is not None:
        data["advantage"] = _safe_serialize(advantage)

    return data


def _convert_trajectory(trajectory: Any) -> list[tuple[SpanType, dict[str, Any]]]:
    """Convert a Trajectory into ``(span_type, data)`` records."""
    records: list[tuple[SpanType, dict[str, Any]]] = []
    uid = getattr(trajectory, "uid", str(uuid.uuid4()))

    # -- trajectory.start --------------------------------------------------
    start: dict[str, Any] = {"uid": uid, "name": getattr(trajectory, "name", "agent")}
    for field in ("task", "metadata", "signals", "input"):
        val = getattr(trajectory, field, None)
        if val is not None and val != {} and val != []:
            start[field] = _safe_serialize(val)
    records.append(("trajectory.start", start))

    # -- trajectory.step (one per step) ------------------------------------
    steps = getattr(trajectory, "steps", [])
    for idx, step in enumerate(steps):
        records.append(("trajectory.step", _extract_step(step, uid, idx)))

    # -- trajectory.end ----------------------------------------------------
    end: dict[str, Any] = {"uid": uid, "num_steps": len(steps)}
    reward = getattr(trajectory, "reward", None)
    if reward is not None:
        end["reward"] = float(reward)
    output = getattr(trajectory, "output", None)
    if output is not None:
        end["output"] = _safe_serialize(output)
    signals = getattr(trajectory, "signals", None)
    if signals:
        end["signals"] = _safe_serialize(signals)
    records.append(("trajectory.end", end))

    return records


def _convert_episode(episode: Any) -> list[tuple[SpanType, dict[str, Any]]]:
    """Convert an Episode into ``(span_type, data)`` records."""
    records: list[tuple[SpanType, dict[str, Any]]] = []

    # -- episode.start -----------------------------------------------------
    ep_start: dict[str, Any] = {"id": getattr(episode, "id", str(uuid.uuid4()))}
    task = getattr(episode, "task", None)
    if task is not None:
        ep_start["task"] = _safe_serialize(task)
    metadata = getattr(episode, "metadata", None)
    if metadata is not None and metadata != {}:
        ep_start["metadata"] = _safe_serialize(metadata)
    records.append(("episode.start", ep_start))

    # -- trajectories ------------------------------------------------------
    trajectories = getattr(episode, "trajectories", [])
    for traj in trajectories:
        records.extend(_convert_trajectory(traj))

    # -- episode.end -------------------------------------------------------
    ep_end: dict[str, Any] = {
        "id": getattr(episode, "id", ""),
        "is_correct": bool(getattr(episode, "is_correct", False)),
        "num_trajectories": len(trajectories),
    }
    termination = getattr(episode, "termination_reason", None)
    if termination is not None:
        ep_end["termination_reason"] = termination.value if hasattr(termination, "value") else str(termination)
    metrics = getattr(episode, "metrics", None)
    if metrics:
        ep_end["metrics"] = _safe_serialize(metrics)
    records.append(("episode.end", ep_end))

    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_trajectories(
    data: Any,
    *,
    output_dir: str | None = None,
) -> str:
    """Export rllm Trajectory or Episode objects to an NDJSON file.

    Args:
        data: A single Trajectory, a list of Trajectories, or an Episode.
            Detected via duck typing (``trajectories`` attr → Episode,
            ``steps`` attr → Trajectory).
        output_dir: Directory for the output file.
            Defaults to ``./trajectories``.

    Returns:
        Absolute path to the written NDJSON file.
    """
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # -- Convert to records ------------------------------------------------
    records: list[tuple[SpanType, dict[str, Any]]] = []

    if hasattr(data, "trajectories"):
        # Episode-like
        records = _convert_episode(data)
    elif isinstance(data, list | tuple):
        # List of trajectories
        for traj in data:
            records.extend(_convert_trajectory(traj))
    elif hasattr(data, "steps"):
        # Single trajectory
        records = _convert_trajectory(data)
    else:
        raise TypeError(f"Expected a Trajectory, list of Trajectories, or Episode, got {type(data).__name__}")

    # -- Write NDJSON ------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{short_id}.ndjson"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        for span_type, span_data in records:
            line = json.dumps({"type": span_type, "data": span_data}, default=str)
            f.write(line + "\n")

    abs_path = os.path.abspath(filepath)
    print(f"Trajectory file saved: {abs_path}")
    return abs_path
