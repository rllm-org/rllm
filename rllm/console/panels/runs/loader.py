"""Filesystem helpers for the Runs panel.

These mirror the helpers ``rllm/eval/visualizer.py`` used to power the
Runs / Episode views. Kept self-contained so the panel can be lifted
elsewhere without dragging in the legacy viewer.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rllm.eval import trace_loader

_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


def resolve_episodes_dir(run_dir: Path) -> Path:
    candidate = run_dir / "episodes"
    return candidate if candidate.is_dir() else run_dir


def load_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_run_timestamp(run_id: str) -> str | None:
    """Pull a ``YYYYMMDD_HHMMSS`` suffix off ``run_id`` and ISO-format it."""
    m = _TIMESTAMP_RE.search(run_id)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def gateway_run_id(run_dir: Path) -> str:
    """Resolve the run's gateway ``run_id`` (meta override → dir basename)."""
    meta = load_meta(run_dir)
    rid = meta.get("gateway_run_id")
    if isinstance(rid, str) and rid:
        return rid
    return run_dir.name


def scan_runs(root: Path) -> list[dict[str, Any]]:
    """Discover all eval runs under ``root``.

    A run is a subdirectory with an ``episodes/`` folder. The aggregate
    ``EvalResult`` JSON sits next to it as ``results.json`` (preferred) or
    legacy ``<root>/<run_id>.json``.
    """
    if not root.is_dir():
        return []

    out: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        episodes_dir = run_dir / "episodes"
        if not episodes_dir.is_dir():
            continue

        run_id = run_dir.name
        meta = load_meta(run_dir)

        agg_path = run_dir / "results.json"
        if not agg_path.is_file():
            legacy = root / f"{run_id}.json"
            agg_path = legacy if legacy.is_file() else agg_path

        agg: dict[str, Any] = {}
        if agg_path.is_file():
            try:
                with open(agg_path, encoding="utf-8") as f:
                    agg = json.load(f)
            except Exception:
                agg = {}

        n_episodes = sum(1 for _ in episodes_dir.glob("episode_*.json"))
        status = "completed" if agg else "incomplete"

        created_at = parse_run_timestamp(run_id)
        if created_at is None:
            try:
                created_at = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()
            except Exception:
                created_at = None

        ts_match = _TIMESTAMP_RE.search(run_id)
        out.append(
            {
                "id": run_id,
                "benchmark": agg.get("dataset_name") or meta.get("benchmark") or "—",
                "model": agg.get("model") or meta.get("model") or "—",
                "agent": agg.get("agent") or meta.get("agent") or "—",
                "split": meta.get("split") or "",
                "timestamp": meta.get("timestamp") or (ts_match.group(1) if ts_match else ""),
                "created_at": created_at,
                "score": agg.get("score"),
                "correct": agg.get("correct"),
                "total": agg.get("total"),
                "errors": agg.get("errors"),
                "n_episodes": n_episodes,
                "status": status,
            }
        )

    out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return out


def load_tasks_jsonl(run_dir: Path) -> list[dict[str, Any]]:
    """Read every line of ``<run_dir>/tasks.jsonl`` (best-effort)."""
    path = run_dir / "tasks.jsonl"
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def finished_eval_idxs(episodes_dir: Path) -> set[int]:
    """Return the set of ``eval_idx`` values for which an episode JSON exists."""
    if not episodes_dir.is_dir():
        return set()
    out: set[int] = set()
    for path in episodes_dir.glob("episode_*.json"):
        m = re.match(r"^episode_(\d+)_", path.name)
        if m:
            try:
                out.add(int(m.group(1)))
            except ValueError:
                continue
    return out


def build_live_payload(run_dir: Path) -> dict[str, Any]:
    """Snapshot of in-flight tasks + completion counts for one run."""
    tasks = load_tasks_jsonl(run_dir)
    finished_idx = finished_eval_idxs(resolve_episodes_dir(run_dir))
    run_id = gateway_run_id(run_dir)
    summaries = trace_loader.session_summaries(trace_loader.default_db_path(), run_id=run_id)
    now = datetime.now(tz=timezone.utc).timestamp()

    in_flight: list[dict[str, Any]] = []
    for t in tasks:
        idx = t.get("idx")
        if not isinstance(idx, int) or idx in finished_idx:
            continue
        sid = t.get("session_id") or ""
        s = summaries.get(sid, {})
        started_at = float(t.get("started_at") or 0.0)
        in_flight.append(
            {
                "idx": idx,
                "session_id": sid,
                "task_id": t.get("task_id"),
                "instruction": t.get("instruction") or "",
                "started_at": started_at,
                "elapsed_s": max(0.0, now - started_at) if started_at else None,
                "trace_count": int(s.get("trace_count") or 0),
                "last_trace_at": s.get("last_at"),
            }
        )
    in_flight.sort(key=lambda r: r["idx"])

    return {
        "in_flight": in_flight,
        "finished_count": len(finished_idx),
        "started_count": len(tasks),
    }


def build_episode_index(episodes_dir: Path) -> list[dict[str, Any]]:
    """Read just the headline fields from every episode file in a run."""
    index: list[dict[str, Any]] = []
    for path in sorted(episodes_dir.glob("episode_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        task = data.get("task") if isinstance(data.get("task"), dict) else {}
        n_steps = sum(len(t.get("steps") or []) for t in (data.get("trajectories") or []))
        rewards = [t.get("reward") for t in (data.get("trajectories") or []) if t.get("reward") is not None]
        avg_reward = sum(rewards) / len(rewards) if rewards else None
        index.append(
            {
                "filename": path.name,
                "eval_idx": data.get("eval_idx"),
                "task_id": task.get("id") if isinstance(task, dict) else None,
                "is_correct": data.get("is_correct"),
                "termination_reason": data.get("termination_reason"),
                "n_trajectories": len(data.get("trajectories") or []),
                "n_steps": n_steps,
                "reward": avg_reward,
                "instruction_preview": _preview(task.get("instruction") if isinstance(task, dict) else None),
            }
        )
    return index


def _preview(value: Any, limit: int = 140) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s[: limit - 1] + "…" if len(s) > limit else s
