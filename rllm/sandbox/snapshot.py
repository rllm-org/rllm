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
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from rllm.env import env_float

if TYPE_CHECKING:
    from rllm.sandbox.protocol import Sandbox
    from rllm.types import Task

logger = logging.getLogger(__name__)

# Backends with no snapshot mechanism — get_sandbox always takes the cold path.
_NO_SNAPSHOT_BACKENDS = {"docker", "local"}

# Default local trust horizon for a snapshot entry (7 days).
_DEFAULT_TTL_HOURS = env_float("RLLM_SNAPSHOT_TTL_HOURS", 168.0)  # set env var: export RLLM_SNAPSHOT_TTL_HOURS=xxx


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


def env_key(backend: str, base_image: str, run_commands: list[str], install_script: str = "") -> str:
    """Content-hash fingerprint of an environment: ``rllm-env-<hash12>``.

    Hashes ``(backend, base_image, RUN block, install script)`` — never
    ``task.id`` — so GRPO group copies share one key and any image/RUN/install
    change yields a new key (a clean miss, never a stale hit). An empty
    ``install_script`` contributes nothing, keeping task-only keys stable.
    Lowercase + dashes: a legal Daytona snapshot name, Modal/Docker-safe.
    """
    parts = [backend, base_image, *run_commands]
    if install_script:
        parts += ["install:", install_script]
    payload = "\n".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"rllm-env-{digest}"


def env_key_for(task: Task, backend: str, install_script: str = "") -> str:
    """Fingerprint a task's environment via the shared image/RUN resolution."""
    from rllm.eval._resolution import _dockerfile_run_commands, _resolve_image

    return env_key(backend, _resolve_image(task, backend), _dockerfile_run_commands(task), install_script)


def install_script_for(agent_flow: object) -> str:
    """The flow's CLI install script, ``""`` when it has none (host flows, bash loops)."""
    fn = getattr(agent_flow, "install_script", None)
    return fn() if callable(fn) else ""


def keys_for_tasks(tasks: list[Task], backend: str | None, install_script: str = "") -> dict[str, Task]:
    """Map distinct env_keys to a representative task (used by ``rllm snapshot create``)."""
    from rllm.eval._resolution import _resolve_backend

    out: dict[str, Task] = {}
    for task in tasks:
        eff = _resolve_backend(task, backend)
        if eff in _NO_SNAPSHOT_BACKENDS:
            continue
        out.setdefault(env_key_for(task, eff, install_script), task)
    return out


def _slug(dataset: str) -> str:
    """Lowercase a dataset name with non-alphanumeric runs collapsed to dashes."""
    return re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-") or "dataset"


def _slice_id_token(slice_spec: dict, task_count: int) -> str:
    """The id fragment that disambiguates a group: ``first{N}`` / ``idx{count}`` / ``all``."""
    kind = slice_spec.get("kind")
    if kind == "max_examples":
        return f"first{slice_spec.get('value')}"
    if kind == "task_indices":
        return f"idx{task_count}"  # number of indices; the raw expression lives in slice["value"]
    return "all"


def _make_group_id(dataset: str, slice_spec: dict, task_count: int) -> str:
    """``slug(dataset)-<slice token>-rand8``. The rand8 is per-invocation, never a content hash."""
    return f"{_slug(dataset)}-{_slice_id_token(slice_spec, task_count)}-{os.urandom(4).hex()}"


def get_sandbox(task: Task, backend: str | None, registry: SnapshotRegistry | None = None, install_script: str = "") -> Sandbox:
    """Return a ready sandbox for ``task`` — from a snapshot when available, else cold.

    Snapshots are used iff ``registry`` is given. Two cheap gates before the cold
    path: a local registry lookup (no network; an absent or expired entry goes
    straight to cold), then an optimistic boot from the stored ref. If that ref
    vanished backend-side the boot raises :class:`SnapshotNotFound` and we fall back
    to cold. Read-only w.r.t. snapshots — building is ``rllm snapshot``'s job.

    ``sandbox.baked_install`` records the install script the booted image
    actually contains: the requested script on an exact-fingerprint hit, ``""``
    when the boot came from the task-only fallback snapshot (better than cold,
    but the caller must still install at runtime). Cold sandboxes leave the
    attribute unset.
    """
    from rllm.eval._resolution import _create_base_sandbox, _create_sandbox_for_task, _resolve_backend
    from rllm.sandbox.protocol import SnapshotNotFound

    backend = _resolve_backend(task, backend)
    if registry is not None and backend not in _NO_SNAPSHOT_BACKENDS:
        key = env_key_for(task, backend, install_script)
        candidates = [key]
        if install_script:
            candidates.append(env_key_for(task, backend))
        for candidate in candidates:
            ref = registry.lookup_env(candidate, backend)
            if ref is None:
                continue
            try:
                sandbox = _create_base_sandbox(task, backend, image=ref)  # snapshot has RUN baked — no replay
            except SnapshotNotFound:
                logger.info("snapshot %s gone on %s — cold fallback", ref, backend)
                registry.discard(candidate)  # so sibling tasks this run skip the doomed boot
                continue
            sandbox.baked_install = install_script if candidate == key else ""
            return sandbox
    return _create_sandbox_for_task(task, backend)


class SnapshotRegistry:
    """Local index of built snapshots at ``~/.rllm/snapshots.json`` (envelope v2).

    Two tables. ``envs[env_key] -> {backend, ref, base_image, created_at, expires_at}``
    is the content layer: one content-addressed, TTL-mortal built artifact per key
    (the backend is the source of truth — :meth:`lookup_env` is pure-local, a miss
    or expired entry means cold with no network call). ``groups[group_id] ->
    {dataset, backend, slice, tasks, created_at, ttl_hours}`` is a thin named pointer
    manifest of one ``create`` invocation; ``tasks`` is the single source of
    membership and a group's env_keys are the unique set over ``tasks[].env_key``.
    Liveness and refcounts are always *derived* by scanning groups over envs, never
    stored.
    """

    def __init__(self, path: str | None = None, envs: dict | None = None, groups: dict | None = None) -> None:
        self.path = path or _registry_path()
        self._envs: dict[str, dict] = envs if envs is not None else {}
        self._groups: dict[str, dict] = groups if groups is not None else {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls) -> SnapshotRegistry:
        path = _registry_path()
        data = cls._read(path)
        return cls(path, envs=data.get("envs", {}), groups=data.get("groups", {}))

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
        """Drop a dead env and persist the prune so the next process skips the doomed boot."""
        with self._lock:
            self._reload_locked()  # merge with on-disk writes before pruning
            if self._envs.pop(key, None) is not None:
                self._save()

    # -- read helpers (pure-local, no live calls) -------------------------

    def env_entries(self) -> dict[str, dict]:
        """The raw ``env_key -> metadata`` table (``list --verbose``)."""
        return dict(self._envs)

    def groups(self) -> dict[str, dict]:
        """The raw ``group_id -> manifest`` table."""
        return dict(self._groups)

    def group_members(self, group_id: str) -> list[str]:
        """The distinct env_keys a group points at (derived from its ``tasks``)."""
        group = self._groups.get(group_id)
        if group is None:
            return []
        return list(dict.fromkeys(t["env_key"] for t in group.get("tasks", [])))

    def groups_referencing(self, env_key: str) -> list[str]:
        """Group ids whose membership includes ``env_key`` (a plain scan)."""
        return [gid for gid in self._groups if env_key in self.group_members(gid)]

    def refcount(self, env_key: str, exclude_group: str | None = None) -> int:
        """How many groups (optionally excluding one) still point at ``env_key``."""
        return sum(gid != exclude_group for gid in self.groups_referencing(env_key))

    # -- mutation (CLI only) ----------------------------------------------

    def record_group(self, dataset: str, backend: str, slice_spec: dict, tasks: list[dict], *, ttl_hours: float = _DEFAULT_TTL_HOURS, force: bool = False) -> str:
        """Record each env once (force-aware) and write a group manifest; return its new id.

        ``tasks`` is a list of ``{id, env_key, ref, base_image}`` for the exact tasks
        this creation covered; the stored group keeps only ``{id, env_key}`` (the
        env layer holds ref/base_image). A fresh random ``group_id`` is minted per
        invocation, so re-running ``create`` honestly records a new creation.
        """
        with self._lock:
            self._reload_locked()
            for t in tasks:
                self._envs[t["env_key"]] = self._env_entry(t["env_key"], backend, t["ref"], t["base_image"], ttl_hours, force)
            group_id = _make_group_id(dataset, slice_spec, len(tasks))
            self._groups[group_id] = {
                "dataset": dataset,
                "backend": backend,
                "slice": slice_spec,
                "tasks": [{"id": t["id"], "env_key": t["env_key"]} for t in tasks],
                "created_at": _now().isoformat(),
                "ttl_hours": ttl_hours,
            }
            self._save()
            return group_id

    def renew(self, group_id: str, ttl_hours: float = _DEFAULT_TTL_HOURS) -> int:
        """Refresh the expiry of a group's members without rebuilding. Returns members renewed."""
        with self._lock:
            self._reload_locked()
            members = self.group_members(group_id)
            renewed = 0
            new_expiry = (_now() + timedelta(hours=ttl_hours)).isoformat()
            for key in members:
                entry = self._envs.get(key)
                if entry is None:
                    continue
                entry["expires_at"] = new_expiry  # explicit refresh — the only non-force path that moves TTL
                renewed += 1
            self._save()
            return renewed

    def destroy_group(self, group_id: str) -> dict:
        """Remove a group, deleting backend refs that no other group still references.

        For each of the group's env_keys: if a *derived* refcount (other groups
        still listing it) is 0, call the backend delete and drop the local env
        only when it returned ``True`` (verified gone). Returns a summary dict.
        """
        from rllm.sandbox.sandboxed_flow import delete_snapshot

        with self._lock:
            self._reload_locked()
            group = self._groups.pop(group_id, None)
            if group is None:
                return {"found": False, "shared": [], "deleted": [], "kept": []}
            members = list(dict.fromkeys(t["env_key"] for t in group.get("tasks", [])))
            backend = group.get("backend")
            shared, deleted, kept = [], [], []
            for key in members:
                entry = self._envs.get(key)
                if entry is None:
                    continue
                if self.refcount(key, exclude_group=group_id) > 0:
                    shared.append(key)  # another group still needs it — leave it be
                elif delete_snapshot(entry.get("backend", backend), entry["ref"]):
                    deleted.append(key)
                    del self._envs[key]  # drop only on confirmed backend deletion
                else:
                    kept.append(key)  # delete unconfirmed — keep so a later run can retry
            self._save()
            return {"found": True, "shared": shared, "deleted": deleted, "kept": kept}

    def sync(self, backend: str) -> dict:
        """Reconcile this backend's local envs against reality — drop only verified-absent ones.

        Per-backend, no-boot probe (Daytona ``snapshot.get``, Modal ``from_id.hydrate``):
        a local env is pruned ONLY on verified backend absence (typed NotFound /
        terminal state), NEVER on auth/permission/rate-limit/timeout. Persisted.
        Returns ``{pruned: [...], kept: <int>}``.
        """
        from rllm.sandbox.sandboxed_flow import snapshot_absent

        with self._lock:
            self._reload_locked()
            keys = [k for k, e in self._envs.items() if e.get("backend") == backend]
            pruned = []
            for key in keys:
                ref = self._envs[key].get("ref")
                if ref and snapshot_absent(backend, ref):
                    del self._envs[key]
                    pruned.append(key)
            self._save()
        return {"pruned": pruned, "kept": len(keys) - len(pruned)}

    # -- internals --------------------------------------------------------

    def _reload_locked(self) -> None:
        """Re-read both tables from disk (call under ``self._lock``)."""
        data = self._read(self.path)
        self._envs = data.get("envs", {})
        self._groups = data.get("groups", {})

    def _env_entry(self, key: str, backend: str, ref: str, base_image: str, ttl_hours: float, force: bool) -> dict:
        """Build an env entry; a non-force reuse keeps the prior expiry so re-creates never renew TTL."""
        prior = self._envs.get(key, {})
        prior_expiry = prior.get("expires_at")
        if force or prior_expiry is None:
            expires_at = (_now() + timedelta(hours=ttl_hours)).isoformat()
        else:
            expires_at = prior_expiry  # reuse: only --force or `renew` moves a horizon forward
        return {
            "backend": backend,
            "ref": ref,
            "base_image": base_image,
            "created_at": prior.get("created_at") or _now().isoformat(),
            "expires_at": expires_at,
        }

    # -- persistence ------------------------------------------------------

    @classmethod
    def _read(cls, path: str) -> dict:
        """Read the registry envelope, migrating v1 in place. Empty on missing/corrupt."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"version": 2, "envs": {}, "groups": {}}
        if data.get("version") == 2:
            data.setdefault("envs", {})
            data.setdefault("groups", {})
            return data
        v2 = cls._migrate_v1_to_v2(data)
        cls._write(path, v2)
        return v2

    @staticmethod
    def _migrate_v1_to_v2(v1: dict) -> dict:
        """Split the v1 flat ``envs[key].datasets[]`` into v2 ``envs`` + ``groups``.

        Each v1 env becomes an ``envs`` entry (created_at backfilled to now). For
        each distinct dataset name across the old ``datasets[]`` lists, synthesize
        one ``slug(dataset)-all-rand8`` group whose tasks carry ``id="(migrated)"``
        — v1 never stored real task ids, so exact provenance is unrecoverable.
        """
        now = _now().isoformat()
        v1_envs = v1.get("envs", {})
        envs: dict[str, dict] = {}
        by_dataset: dict[tuple[str, str], list[str]] = {}
        for key, entry in v1_envs.items():
            backend = entry.get("backend")
            envs[key] = {
                "backend": backend,
                "ref": entry.get("ref"),
                "base_image": entry.get("base_image"),
                "created_at": entry.get("created_at") or now,
                "expires_at": entry.get("expires_at"),
            }
            for dataset in entry.get("datasets", []):
                by_dataset.setdefault((dataset, backend), []).append(key)
        groups: dict[str, dict] = {}
        for (dataset, backend), keys in by_dataset.items():
            slice_spec = {"kind": "all", "value": None}
            groups[_make_group_id(dataset, slice_spec, len(keys))] = {
                "dataset": dataset,
                "backend": backend,
                "slice": slice_spec,
                "tasks": [{"id": "(migrated)", "env_key": k} for k in keys],
                "created_at": "(migrated)",
                "ttl_hours": _DEFAULT_TTL_HOURS,
            }
        logger.info("Migrated snapshot registry from v1 to v2 (%d envs, %d groups).", len(envs), len(groups))
        return {"version": 2, "envs": envs, "groups": groups}

    @staticmethod
    def _write(path: str, data: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _save(self) -> None:
        self._write(self.path, {"version": 2, "envs": self._envs, "groups": self._groups})


__all__ = ["env_key", "env_key_for", "install_script_for", "keys_for_tasks", "get_sandbox", "SnapshotRegistry"]
