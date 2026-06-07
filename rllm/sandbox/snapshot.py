"""Environment snapshots for cold-start acceleration.

A snapshot bakes a task's base image + Dockerfile ``RUN`` steps into a backend
artifact keyed by :func:`env_key`. :func:`get_sandbox` boots from one when present
(fast) or creates a cold sandbox otherwise; snapshots are built/destroyed only by
``rllm snapshot``, never by a run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllm.sandbox.protocol import Sandbox
    from rllm.types import Task

logger = logging.getLogger(__name__)

# Backends with no snapshot mechanism — get_sandbox always takes the cold path.
_NO_SNAPSHOT_BACKENDS = {"docker", "local"}

# Default local trust horizon for a snapshot entry (7 days).
_DEFAULT_TTL_HOURS = 168.0


def _registry_path() -> str:
    rllm_home = os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm"))
    return os.path.join(rllm_home, "snapshots.json")


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _expired(iso: str | None) -> bool:
    """True if the ISO timestamp is in the past. Coerces a naive value to UTC."""
    if not iso:
        return False
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return _now() >= dt


def env_key(backend: str, base_image: str, run_commands: list[str]) -> str:
    """Content-hash fingerprint of an environment: ``rllm-env-<hash12>``.

    Hashes only ``(backend, base_image, RUN block)`` — never ``task.id`` — so
    GRPO group copies share one key and any image/RUN change yields a new key
    (a clean miss, never a stale hit). Lowercase + dashes: a legal Daytona
    snapshot name, Modal/Docker-safe.
    """
    payload = "\n".join([backend, base_image, *run_commands])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"rllm-env-{digest}"


def env_key_for(task: Task, backend: str) -> str:
    """Fingerprint a task's environment via the shared image/RUN resolution."""
    from rllm.eval._resolution import _dockerfile_run_commands, _resolve_image

    return env_key(backend, _resolve_image(task, backend), _dockerfile_run_commands(task))


def keys_for_tasks(tasks: list[Task], backend: str | None) -> dict[str, Task]:
    """Map distinct env_keys to a representative task (used by ``rllm snapshot create``)."""
    from rllm.eval._resolution import _resolve_backend

    out: dict[str, Task] = {}
    for task in tasks:
        eff = _resolve_backend(task, backend)
        if eff in _NO_SNAPSHOT_BACKENDS:
            continue
        out.setdefault(env_key_for(task, eff), task)
    return out


def get_sandbox(task: Task, backend: str | None, registry: SnapshotRegistry | None = None) -> Sandbox:
    """Return a ready sandbox for ``task`` — from a snapshot when available, else cold.

    Snapshots are used iff ``registry`` is given. Two cheap gates before the cold
    path: a local registry lookup (no network; an absent or expired entry goes
    straight to cold), then an optimistic boot from the stored ref. If that ref
    vanished backend-side the boot raises :class:`SnapshotNotFound` and we fall back
    to cold. Read-only w.r.t. snapshots — building is ``rllm snapshot``'s job.
    """
    from rllm.eval._resolution import _create_base_sandbox, _create_sandbox_for_task, _resolve_backend
    from rllm.sandbox.protocol import SnapshotNotFound

    backend = _resolve_backend(task, backend)
    if registry is not None and backend not in _NO_SNAPSHOT_BACKENDS:
        key = env_key_for(task, backend)
        ref = registry.lookup_env(key, backend)
        if ref is not None:
            try:
                return _create_base_sandbox(task, backend, image=ref)  # snapshot has RUN baked — no replay
            except SnapshotNotFound:
                logger.info("snapshot %s gone on %s — cold fallback", ref, backend)
                registry.discard(key)  # so sibling tasks this run skip the doomed boot
    return _create_sandbox_for_task(task, backend)


class SnapshotRegistry:
    """Local index of built snapshots at ``~/.rllm/snapshots.json``.

    One flat ``env_key -> {backend, ref, base_image, expires_at, datasets}``
    table; ``datasets`` back-references the collections that requested each env
    so the CLI can list/destroy by dataset. The backend is the source of truth —
    :meth:`lookup_env` is pure-local (a miss or expired entry means cold, no
    network call).
    """

    def __init__(self, path: str | None = None, envs: dict | None = None) -> None:
        self.path = path or _registry_path()
        self._envs: dict[str, dict] = envs if envs is not None else {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls) -> SnapshotRegistry:
        path = _registry_path()
        return cls(path, envs=cls._read(path).get("envs", {}))

    # -- hot path (read-only, no live calls) ------------------------------

    def lookup_env(self, key: str, backend: str) -> str | None:
        """Return a locally-trusted snapshot ref, or ``None`` (→ cold path)."""
        entry = self._envs.get(key)
        if entry is None or entry.get("backend") != backend:
            return None
        if _expired(entry.get("expires_at")):
            return None
        return entry.get("ref")

    def discard(self, key: str) -> None:
        """Drop an env from the in-memory view (after a self-healed cold fallback)."""
        with self._lock:
            self._envs.pop(key, None)

    # -- mutation (CLI only) ----------------------------------------------

    def record_env(self, key: str, backend: str, ref: str, base_image: str, dataset: str | None, *, ttl_hours: float = _DEFAULT_TTL_HOURS) -> None:
        """Add or refresh an env entry and attach ``dataset`` to its back-reference."""
        with self._lock:
            self._envs = self._read(self.path).get("envs", {})  # re-read to merge with on-disk writes since load
            datasets = set(self._envs.get(key, {}).get("datasets", []))
            if dataset:
                datasets.add(dataset)
            self._envs[key] = {
                "backend": backend,
                "ref": ref,
                "base_image": base_image,
                "expires_at": (_now() + timedelta(hours=ttl_hours)).isoformat(),
                "datasets": sorted(datasets),
            }
            self._save()

    def destroy(self, dataset: str, backend: str | None = None) -> tuple[int, int]:
        """Detach ``dataset`` from its envs, deleting backend refs that become orphaned.

        Returns ``(envs_detached, backend_refs_deleted)``.
        """
        from rllm.sandbox.sandboxed_flow import delete_snapshot

        detached = deleted = 0
        with self._lock:
            self._envs = self._read(self.path).get("envs", {})
            for key in list(self._envs):
                entry = self._envs[key]
                if backend and entry.get("backend") != backend:
                    continue
                datasets = [d for d in entry.get("datasets", []) if d != dataset]
                if len(datasets) == len(entry.get("datasets", [])):
                    continue
                detached += 1
                if datasets:
                    entry["datasets"] = datasets
                else:
                    if delete_snapshot(entry["backend"], entry["ref"]):
                        deleted += 1
                    del self._envs[key]
            self._save()
        return detached, deleted

    def collections(self) -> list[dict]:
        """Group envs by ``(dataset, backend)`` for the ``list`` view."""
        groups: dict[tuple[str, str], list[dict]] = {}
        for entry in self._envs.values():
            for dataset in entry.get("datasets", []) or ["(detached)"]:
                groups.setdefault((dataset, entry["backend"]), []).append(entry)
        out = []
        for (dataset, backend), entries in sorted(groups.items()):
            expiries = [e["expires_at"] for e in entries if e.get("expires_at")]
            out.append({"dataset": dataset, "backend": backend, "envs": len(entries), "expires_at": min(expiries) if expiries else None})
        return out

    def env_entries(self) -> dict[str, dict]:
        """The raw ``env_key -> metadata`` table (``list --verbose``)."""
        return dict(self._envs)

    # -- persistence ------------------------------------------------------

    @staticmethod
    def _read(path: str) -> dict:
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 1, "envs": self._envs}, f, indent=2)
        os.replace(tmp, self.path)


__all__ = ["env_key", "env_key_for", "keys_for_tasks", "get_sandbox", "SnapshotRegistry"]
