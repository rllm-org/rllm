"""Filesystem + gateway-store helpers for the Runs panel.

Run summaries are a union of two sources:

* Disk: ``eval_results_root/<run_id>/`` with ``episodes/`` and a final
  aggregate ``results.json``. This is the source for completed-run
  metrics (score, correctness, reward).
* Gateway: the ``runs`` table in ``~/.rllm/gateway/traces.db`` —
  authoritative for liveness (``ended_at IS NULL`` ⇒ in-flight) and
  for session-level trace counts during the run.

The two are joined by ``run_id`` (eval CLI uses the eval-results dir
basename as the gateway ``run_id``; ``meta.json`` may override).
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


def _disk_run_row(run_dir: Path, root: Path) -> dict[str, Any] | None:
    """Build a run-summary row from disk for a single run dir.

    Returns ``None`` when the dir doesn't look like an eval run (no
    ``episodes/``).
    """
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.is_dir():
        return None

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

    created_at = parse_run_timestamp(run_id)
    if created_at is None:
        try:
            created_at = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()
        except Exception:
            created_at = None

    ts_match = _TIMESTAMP_RE.search(run_id)
    return {
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
        "status": "completed" if agg else "incomplete",
        # Set by ``scan_runs`` after merging in gateway-side data.
        "in_flight": False,
        "started_at": None,
        "ended_at": None,
    }


def scan_runs(root: Path) -> list[dict[str, Any]]:
    """Union of disk + gateway runs, ordered by start time DESC.

    A run is "in-flight" when the gateway's ``runs`` row has
    ``ended_at IS NULL``. A run with disk data but no gateway row reads
    as completed (legacy / older eval dirs predate the runs table).
    """
    by_id: dict[str, dict[str, Any]] = {}

    if root.is_dir():
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            row = _disk_run_row(run_dir, root)
            if row is None:
                continue
            by_id[row["id"]] = row

    db = trace_loader.default_db_path()
    for gw in trace_loader.list_runs(db):
        rid = gw["run_id"]
        existing = by_id.get(rid)
        in_flight = gw.get("ended_at") is None
        started_at = gw.get("started_at")
        ended_at = gw.get("ended_at")
        if existing is not None:
            existing["in_flight"] = in_flight
            existing["started_at"] = started_at
            existing["ended_at"] = ended_at
            # Status flips to "running" when the gateway says it's live
            # but disk hasn't seen results.json yet.
            if in_flight and existing["status"] == "incomplete":
                existing["status"] = "running"
            continue
        # Gateway-only run (no disk dir yet — usually a fresh in-flight run).
        meta = gw.get("metadata") or {}
        by_id[rid] = {
            "id": rid,
            "benchmark": meta.get("benchmark") or "—",
            "model": meta.get("model") or "—",
            "agent": meta.get("agent") or "—",
            "split": meta.get("split") or "",
            "timestamp": "",
            "created_at": _ts_to_iso(started_at),
            "score": None,
            "correct": None,
            "total": None,
            "errors": None,
            "n_episodes": 0,
            "status": "running" if in_flight else "completed",
            "in_flight": in_flight,
            "started_at": started_at,
            "ended_at": ended_at,
        }

    out = list(by_id.values())
    out.sort(
        key=lambda r: (
            r.get("started_at") or 0.0,
            r.get("created_at") or "",
        ),
        reverse=True,
    )
    return out


def _ts_to_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


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
    """Liveness snapshot for one run.

    Liveness comes from the gateway's ``runs`` row (started_at /
    ended_at). Per-session trace counts come from the trace store.
    Per-task instructions still come from ``tasks.jsonl`` so the live
    banner can show what each in-flight session is working on.
    """
    db = trace_loader.default_db_path()
    rid = gateway_run_id(run_dir)

    gateway_rows = trace_loader.list_runs(db)
    gateway_row = next((r for r in gateway_rows if r["run_id"] == rid), None)
    started_at = gateway_row.get("started_at") if gateway_row else None
    ended_at = gateway_row.get("ended_at") if gateway_row else None
    in_flight = gateway_row is not None and ended_at is None

    sessions = [
        {
            "session_id": sid,
            "trace_count": int(s.get("trace_count") or 0),
            "first_at": s.get("first_at"),
            "last_at": s.get("last_at"),
        }
        for sid, s in trace_loader.session_summaries(db, run_id=rid).items()
    ]
    sessions.sort(key=lambda r: r.get("last_at") or 0.0, reverse=True)

    tasks = load_tasks_jsonl(run_dir)
    finished_idx = finished_eval_idxs(resolve_episodes_dir(run_dir))
    summaries_by_session = {s["session_id"]: s for s in sessions}
    now = datetime.now(tz=timezone.utc).timestamp()

    in_flight_tasks: list[dict[str, Any]] = []
    for t in tasks:
        idx = t.get("idx")
        if not isinstance(idx, int) or idx in finished_idx:
            continue
        sid = t.get("session_id") or ""
        s = summaries_by_session.get(sid, {})
        task_started = float(t.get("started_at") or 0.0)
        in_flight_tasks.append(
            {
                "idx": idx,
                "session_id": sid,
                "task_id": t.get("task_id"),
                "instruction": t.get("instruction") or "",
                "started_at": task_started,
                "elapsed_s": max(0.0, now - task_started) if task_started else None,
                "trace_count": int(s.get("trace_count") or 0),
                "last_trace_at": s.get("last_at"),
            }
        )
    in_flight_tasks.sort(key=lambda r: r["idx"])

    return {
        "run_id": rid,
        "started_at": started_at,
        "ended_at": ended_at,
        "in_flight": in_flight,
        "sessions": sessions,
        "in_flight_tasks": in_flight_tasks,
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
