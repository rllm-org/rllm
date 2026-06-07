"""Offline tests for the snapshot system (env_key, get_sandbox, SnapshotRegistry).

No cloud backend is touched: sandbox creation is monkeypatched so the *logic* —
env_key/GRPO invariance, the two-gate get_sandbox (local lookup → optimistic
boot → cold self-heal), the no-live-call-on-miss rule, and the flat registry's
persistence / dataset collections / refcount destroy — is exercised
deterministically.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rllm.sandbox.protocol import SnapshotNotFound
from rllm.sandbox.snapshot import SnapshotRegistry, env_key, env_key_for, get_sandbox, keys_for_tasks
from rllm.types import Task


def _task(id="task-1", image="swebench/sweb.eval.x86_64.foo:latest", backend="modal"):
    return Task(
        id=id,
        instruction="do the thing",
        metadata={"environment": {"docker_image": image}, "sandbox_backend": backend},
        dataset_dir=Path("/nonexistent"),
        sub_dir=None,
    )


def _future() -> str:
    return (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()


def _past() -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()


class _FakeSandbox:
    def close(self):
        pass


# --------------------------------------------------------------------------
# env_key — identity / GRPO invariance / sensitivity
# --------------------------------------------------------------------------


def test_env_key_sensitive_and_legal_name():
    base = env_key("modal", "img:tag", ["run a"])
    assert env_key("daytona", "img:tag", ["run a"]) != base  # backend
    assert env_key("modal", "img:other", ["run a"]) != base  # base image
    assert env_key("modal", "img:tag", ["run b"]) != base  # RUN block
    # Legal Daytona snapshot name: lowercase + dashes only.
    assert base.startswith("rllm-env-") and base.replace("-", "").isalnum() and base.islower()


def test_env_key_independent_of_task_id():
    """GRPO guarantee: group copies differing only by id share one snapshot."""
    a = _task(id="uuid-aaaa")
    b = _task(id="uuid-bbbb")
    assert env_key_for(a, "modal") == env_key_for(b, "modal")


def test_keys_for_tasks_dedups_envs_and_skips_local_backends():
    same_a, same_b = _task(id="1", image="img:a"), _task(id="2", image="img:a")
    other = _task(id="3", image="img:b")
    assert len(keys_for_tasks([same_a, same_b, other], "modal")) == 2  # a/b collapse
    assert keys_for_tasks([_task(backend="docker")], "docker") == {}  # no snapshots for docker


# --------------------------------------------------------------------------
# get_sandbox — two gates + self-heal + no-live-call-on-miss
# --------------------------------------------------------------------------


def test_get_sandbox_local_hit_boots_from_snapshot(monkeypatch, tmp_path):
    import rllm.eval._resolution as res

    reg = SnapshotRegistry(str(tmp_path / "s.json"))
    reg._envs[env_key_for(_task(), "modal")] = {"backend": "modal", "ref": "im-abc", "expires_at": _future()}

    booted = {}
    monkeypatch.setattr(res, "_create_base_sandbox", lambda task, backend, *, image=None, name=None: booted.update(image=image) or _FakeSandbox())
    monkeypatch.setattr(res, "_create_sandbox_for_task", lambda task, backend: pytest.fail("cold path taken on a local hit"))

    sb = get_sandbox(_task(), "modal", reg)
    assert isinstance(sb, _FakeSandbox)
    assert booted["image"] == "im-abc"  # booted from the ref, no replay


def test_get_sandbox_snapshot_gone_self_heals_to_cold(monkeypatch, tmp_path):
    import rllm.eval._resolution as res

    reg = SnapshotRegistry(str(tmp_path / "s.json"))
    key = env_key_for(_task(), "modal")
    reg._envs[key] = {"backend": "modal", "ref": "im-gone", "expires_at": _future()}

    def _gone(task, backend, *, image=None, name=None):
        raise SnapshotNotFound("gone")

    cold = {}
    monkeypatch.setattr(res, "_create_base_sandbox", _gone)
    monkeypatch.setattr(res, "_create_sandbox_for_task", lambda task, backend: cold.update(cold=True) or _FakeSandbox())

    assert isinstance(get_sandbox(_task(), "modal", reg), _FakeSandbox)
    assert cold["cold"]
    assert reg.lookup_env(key, "modal") is None  # dead entry pruned so siblings skip the doomed boot


def test_get_sandbox_local_miss_is_cold_with_no_live_call(monkeypatch, tmp_path):
    import rllm.eval._resolution as res

    reg = SnapshotRegistry(str(tmp_path / "s.json"))  # empty → miss
    cold = {}
    monkeypatch.setattr(res, "_create_base_sandbox", lambda *a, **k: pytest.fail("snapshot boot attempted on a local miss"))
    monkeypatch.setattr(res, "_create_sandbox_for_task", lambda task, backend: cold.update(cold=True) or _FakeSandbox())

    assert isinstance(get_sandbox(_task(), "modal", reg), _FakeSandbox)
    assert cold["cold"]


def test_get_sandbox_no_snapshot_backend_never_consults_registry(monkeypatch, tmp_path):
    import rllm.eval._resolution as res

    reg = SnapshotRegistry(str(tmp_path / "s.json"))
    monkeypatch.setattr(res, "_create_base_sandbox", lambda *a, **k: pytest.fail("docker has no snapshot path"))
    monkeypatch.setattr(res, "_create_sandbox_for_task", lambda task, backend: _FakeSandbox())
    assert isinstance(get_sandbox(_task(backend="docker"), "docker", reg), _FakeSandbox)


def test_get_sandbox_no_registry_forces_cold(monkeypatch):
    import rllm.eval._resolution as res

    # registry=None (what --no-snapshot passes) must never touch the snapshot path.
    monkeypatch.setattr(res, "_create_base_sandbox", lambda *a, **k: pytest.fail("no registry must not boot from snapshot"))
    monkeypatch.setattr(res, "_create_sandbox_for_task", lambda task, backend: _FakeSandbox())
    assert isinstance(get_sandbox(_task(), "modal", None), _FakeSandbox)


# --------------------------------------------------------------------------
# SnapshotRegistry — lookup expiry, persistence, collections, refcount destroy
# --------------------------------------------------------------------------


def test_lookup_env_absent_expired_mismatch_live(tmp_path):
    reg = SnapshotRegistry(str(tmp_path / "s.json"))
    assert reg.lookup_env("missing", "modal") is None  # absent
    reg._envs["k"] = {"backend": "modal", "ref": "im-1", "expires_at": _past()}
    assert reg.lookup_env("k", "modal") is None  # expired → no live call
    assert reg.lookup_env("k", "daytona") is None  # backend mismatch
    reg._envs["k"]["expires_at"] = _future()
    assert reg.lookup_env("k", "modal") == "im-1"  # live


def test_record_persists_and_groups_by_dataset(monkeypatch, tmp_path):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    reg.record_env("rllm-env-a", "modal", "im-1", "python:3.11", "swebench")
    reg.record_env("rllm-env-b", "modal", "im-2", "python:3.11", "swebench")
    reg.record_env("rllm-env-a", "modal", "im-1", "python:3.11", "gsm8k")  # env shared by 2 datasets

    assert Path(tmp_path / "snapshots.json").exists()
    cols = {(c["dataset"], c["backend"]): c for c in SnapshotRegistry.load().collections()}
    assert cols[("swebench", "modal")]["envs"] == 2
    assert cols[("gsm8k", "modal")]["envs"] == 1
    # round-trip: a fresh load can boot from a recorded ref.
    assert SnapshotRegistry.load().lookup_env("rllm-env-a", "modal") == "im-1"


def test_destroy_refcounts_shared_envs(monkeypatch, tmp_path):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    reg.record_env("rllm-env-a", "modal", "im-1", "img", "swebench")
    reg.record_env("rllm-env-b", "modal", "im-2", "img", "swebench")
    reg.record_env("rllm-env-a", "modal", "im-1", "img", "gsm8k")

    import rllm.sandbox.sandboxed_flow as sf

    deleted: list[str] = []
    monkeypatch.setattr(sf, "delete_snapshot", lambda backend, ref: deleted.append(ref) or True)

    # gsm8k goes away but env-a is still referenced by swebench → not deleted.
    detached, n = SnapshotRegistry.load().destroy("gsm8k")
    assert (detached, n) == (1, 0) and deleted == []

    # swebench goes away → env-b orphaned, env-a now orphaned → both deleted.
    detached, n = SnapshotRegistry.load().destroy("swebench")
    assert detached == 2 and n == 2 and sorted(deleted) == ["im-1", "im-2"]
    assert SnapshotRegistry.load().collections() == []
