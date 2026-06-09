"""Offline tests for the snapshot system (env_key, get_sandbox, SnapshotRegistry v2).

No cloud backend is touched: sandbox creation and the backend probes/deletes are
monkeypatched so the *logic* — env_key/GRPO invariance, the two-gate get_sandbox
(local lookup → optimistic boot → cold self-heal), the no-live-call-on-miss rule,
and the v2 registry's persistence / v1→v2 migration / derived-refcount destroy /
verified-absence sync — is exercised deterministically.
"""

from __future__ import annotations

import json
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


def make_task_env(env_key: str, ref: str, *, base_image="img") -> dict:
    """Build one per-task env descriptor (the shape record_group consumes per task)."""
    return {"id": f"task-{env_key}", "env_key": env_key, "ref": ref, "base_image": base_image}


def make_group(reg: SnapshotRegistry, dataset: str, task_envs: list[dict], *, backend="modal", slice_spec=None, ttl_hours=168.0, force=False) -> str:
    """Record a v2 group through the real record_group path; return its group_id."""
    spec = slice_spec or {"kind": "all", "value": None}
    return reg.record_group(dataset, backend, spec, task_envs, ttl_hours=ttl_hours, force=force)


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
# SnapshotRegistry v2 — lookup expiry, persistence, groups, derived-refcount destroy
# --------------------------------------------------------------------------


def test_lookup_env_absent_expired_mismatch_live(tmp_path):
    reg = SnapshotRegistry(str(tmp_path / "s.json"))
    assert reg.lookup_env("missing", "modal") is None  # absent
    reg._envs["k"] = {"backend": "modal", "ref": "im-1", "expires_at": _past()}
    assert reg.lookup_env("k", "modal") is None  # expired → no live call
    assert reg.lookup_env("k", "daytona") is None  # backend mismatch
    reg._envs["k"]["expires_at"] = _future()
    assert reg.lookup_env("k", "modal") == "im-1"  # live


def test_record_group_persists_and_derives_membership(monkeypatch, tmp_path):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    gid = make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1"), make_task_env("rllm-env-b", "im-2")])

    assert Path(tmp_path / "snapshots.json").exists()
    reloaded = SnapshotRegistry.load()
    assert sorted(reloaded.group_members(gid)) == ["rllm-env-a", "rllm-env-b"]
    # round-trip: a fresh load can boot from a recorded ref.
    assert reloaded.lookup_env("rllm-env-a", "modal") == "im-1"
    # the group records the exact tasks it was created from (D3).
    tasks = reloaded.groups()[gid]["tasks"]
    assert {t["env_key"] for t in tasks} == {"rllm-env-a", "rllm-env-b"}


def test_group_id_shape_and_randomness(monkeypatch, tmp_path):
    """slug-slicelabel-rand8; re-running create mints a NEW id (rand8 is per-invocation)."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    a = make_group(reg, "harbor:swebench-verified", [make_task_env("rllm-env-a", "im-1")], slice_spec={"kind": "max_examples", "value": 5})
    b = make_group(reg, "harbor:swebench-verified", [make_task_env("rllm-env-a", "im-1")], slice_spec={"kind": "max_examples", "value": 5})
    assert a.startswith("harbor-swebench-verified-first5-") and b.startswith("harbor-swebench-verified-first5-")
    assert a != b  # same request content, different creation → distinct ids


def test_refcount_and_destroy_group_shared_survives(monkeypatch, tmp_path):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    g1 = make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1"), make_task_env("rllm-env-b", "im-2")])
    g2 = make_group(reg, "gsm8k", [make_task_env("rllm-env-a", "im-1")])  # shares env-a

    import rllm.sandbox.sandboxed_flow as sf

    deleted: list[str] = []
    monkeypatch.setattr(sf, "delete_snapshot", lambda backend, ref: deleted.append(ref) or True)

    # destroy g1: env-b is unshared → deleted; env-a still in g2 → survives.
    res = SnapshotRegistry.load().destroy_group(g1)
    assert res["found"] and res["deleted"] == ["rllm-env-b"] and res["shared"] == ["rllm-env-a"]
    assert deleted == ["im-2"]
    reloaded = SnapshotRegistry.load()
    assert "rllm-env-a" in reloaded.env_entries() and "rllm-env-b" not in reloaded.env_entries()
    assert g1 not in reloaded.groups() and g2 in reloaded.groups()

    # now destroy g2: env-a is unshared → deleted.
    res = SnapshotRegistry.load().destroy_group(g2)
    assert res["deleted"] == ["rllm-env-a"] and sorted(deleted) == ["im-1", "im-2"]
    assert SnapshotRegistry.load().env_entries() == {}


def test_destroy_group_keeps_env_when_delete_unconfirmed(monkeypatch, tmp_path):
    """An unconfirmed backend delete (auth-scoped key) must KEEP the local env for retry."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    g = make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1")])

    import rllm.sandbox.sandboxed_flow as sf

    monkeypatch.setattr(sf, "delete_snapshot", lambda backend, ref: False)  # keep
    res = SnapshotRegistry.load().destroy_group(g)
    assert res["found"] and res["kept"] == ["rllm-env-a"] and res["deleted"] == []
    reloaded = SnapshotRegistry.load()
    assert g not in reloaded.groups()  # group removed
    assert reloaded.env_entries()["rllm-env-a"]["ref"] == "im-1"  # env kept — ref didn't leak silently


def test_sync_drops_only_verified_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1"), make_task_env("rllm-env-b", "im-2")])

    import rllm.sandbox.sandboxed_flow as sf

    # im-1 verifiably gone, im-2 present.
    monkeypatch.setattr(sf, "snapshot_absent", lambda backend, ref: ref == "im-1")
    res = SnapshotRegistry.load().sync("modal")
    assert res["pruned"] == ["rllm-env-a"] and res["kept"] == 1
    assert "rllm-env-a" not in SnapshotRegistry.load().env_entries()


def test_sync_keeps_all_on_auth_failure(monkeypatch, tmp_path):
    """A sync that can't confirm absence (auth/permission → snapshot_absent False) prunes nothing."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1"), make_task_env("rllm-env-b", "im-2")])

    import rllm.sandbox.sandboxed_flow as sf

    monkeypatch.setattr(sf, "snapshot_absent", lambda backend, ref: False)  # never confirmed gone
    res = SnapshotRegistry.load().sync("modal")
    assert res["pruned"] == [] and res["kept"] == 2
    assert set(SnapshotRegistry.load().env_entries()) == {"rllm-env-a", "rllm-env-b"}


def test_discard_persists_across_processes(monkeypatch, tmp_path):
    """discard must rewrite the file so the next process doesn't reload the dead entry."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1")])

    reg.discard("rllm-env-a")
    assert SnapshotRegistry.load().lookup_env("rllm-env-a", "modal") is None  # gone from disk too


def test_reuse_does_not_renew_ttl(monkeypatch, tmp_path):
    """A non-force re-create reuses the env and keeps its horizon — even with a longer ttl (no immortality)."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1")], ttl_hours=1.0)
    first = SnapshotRegistry.load().env_entries()["rllm-env-a"]["expires_at"]

    # re-create non-force with the DEFAULT (longer) ttl must NOT push the horizon out.
    make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1")], ttl_hours=168.0)
    assert SnapshotRegistry.load().env_entries()["rllm-env-a"]["expires_at"] == first


def test_force_refreshes_ttl(monkeypatch, tmp_path):
    """--force is the explicit rebuild path that moves the horizon forward."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1")], ttl_hours=1.0)
    short = SnapshotRegistry.load().env_entries()["rllm-env-a"]["expires_at"]

    make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1")], ttl_hours=168.0, force=True)
    assert SnapshotRegistry.load().env_entries()["rllm-env-a"]["expires_at"] > short  # horizon moved forward


def test_renew_refreshes_member_ttls_without_rebuild(monkeypatch, tmp_path):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    reg._envs = {"rllm-env-a": {"backend": "modal", "ref": "im-1", "base_image": "img", "created_at": _past(), "expires_at": _past()}}
    reg._groups = {
        "swebench-all-deadbeef": {
            "dataset": "swebench",
            "backend": "modal",
            "slice": {"kind": "all", "value": None},
            "tasks": [{"id": "t", "env_key": "rllm-env-a"}],
            "created_at": _past(),
            "ttl_hours": 168.0,
        }
    }
    reg._save()

    renewed = reg.renew("swebench-all-deadbeef", 168.0)
    assert renewed == 1
    assert datetime.fromisoformat(SnapshotRegistry.load().env_entries()["rllm-env-a"]["expires_at"]) > datetime.now(tz=timezone.utc)


def test_created_at_set_once_and_preserved(monkeypatch, tmp_path):
    """created_at is stamped on first build and preserved across reuse and force rebuild."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    reg = SnapshotRegistry.load()
    make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1")])
    created = SnapshotRegistry.load().env_entries()["rllm-env-a"]["created_at"]
    assert created  # set on first build

    make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-1")])  # reuse
    make_group(reg, "swebench", [make_task_env("rllm-env-a", "im-2")], force=True)  # force rebuild
    assert SnapshotRegistry.load().env_entries()["rllm-env-a"]["created_at"] == created


def test_migrate_v1_to_v2_splits_envs_and_synthesizes_groups(monkeypatch, tmp_path):
    """A v1 flat datasets[] registry migrates in place to envs + per-dataset (migrated) groups."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    path = tmp_path / "snapshots.json"
    v1 = {
        "version": 1,
        "envs": {
            "rllm-env-a": {"backend": "modal", "ref": "im-1", "base_image": "img", "expires_at": _future(), "datasets": ["swebench", "gsm8k"]},
            "rllm-env-b": {"backend": "modal", "ref": "im-2", "base_image": "img", "expires_at": _future(), "datasets": ["swebench"]},
        },
    }
    path.write_text(json.dumps(v1))

    reg = SnapshotRegistry.load()
    on_disk = json.loads(path.read_text())
    assert on_disk["version"] == 2  # migrated in place on load
    # envs carry no datasets[] anymore; created_at backfilled.
    assert "datasets" not in reg.env_entries()["rllm-env-a"]
    assert reg.env_entries()["rllm-env-a"]["created_at"]

    groups = reg.groups()
    assert len(groups) == 2  # one per distinct dataset name
    by_dataset = {g["dataset"]: g for g in groups.values()}
    assert sorted(by_dataset) == ["gsm8k", "swebench"]
    assert by_dataset["swebench"]["slice"]["kind"] == "all"
    assert by_dataset["swebench"]["created_at"] == "(migrated)"  # honest: v1 had no real date
    assert {t["env_key"] for t in by_dataset["swebench"]["tasks"]} == {"rllm-env-a", "rllm-env-b"}
    assert all(t["id"] == "(migrated)" for g in groups.values() for t in g["tasks"])  # no real task ids in v1


# --------------------------------------------------------------------------
# Modal check-before-build — reuse when the (mocked) probe says alive, rebuild on NotFound
# --------------------------------------------------------------------------


def test_modal_build_reuses_when_prior_ref_alive(monkeypatch):
    import rllm.sandbox.backends.modal_backend as mb

    monkeypatch.setattr(mb, "_modal_ref_alive", lambda ref: True)
    # a live prior ref must short-circuit before any sandbox build.
    import rllm.eval._resolution as res

    monkeypatch.setattr(res, "_create_base_sandbox", lambda *a, **k: pytest.fail("rebuilt despite a live prior ref"))
    assert mb.build_modal_snapshot(_task(), "rllm-env-a", prior_ref="im-old") == "im-old"


def test_modal_build_rebuilds_when_prior_ref_gone(monkeypatch):
    import rllm.eval._resolution as res
    import rllm.sandbox.backends.modal_backend as mb

    monkeypatch.setattr(mb, "_modal_ref_alive", lambda ref: False)  # NotFound → rebuild

    class _Img:
        object_id = "im-new"

    class _SB:
        _sandbox = type("S", (), {"snapshot_filesystem": staticmethod(lambda: _Img())})()

        def close(self):
            pass

    monkeypatch.setattr(res, "_create_base_sandbox", lambda *a, **k: _SB())
    monkeypatch.setattr(res, "_replay_dockerfile", lambda *a, **k: None)
    assert mb.build_modal_snapshot(_task(), "rllm-env-a", prior_ref="im-old") == "im-new"


def test_modal_build_force_rebuilds_even_when_alive(monkeypatch):
    import rllm.eval._resolution as res
    import rllm.sandbox.backends.modal_backend as mb

    monkeypatch.setattr(mb, "_modal_ref_alive", lambda ref: pytest.fail("probe should be skipped under force"))

    class _Img:
        object_id = "im-new"

    class _SB:
        _sandbox = type("S", (), {"snapshot_filesystem": staticmethod(lambda: _Img())})()

        def close(self):
            pass

    monkeypatch.setattr(res, "_create_base_sandbox", lambda *a, **k: _SB())
    monkeypatch.setattr(res, "_replay_dockerfile", lambda *a, **k: None)
    assert mb.build_modal_snapshot(_task(), "rllm-env-a", prior_ref="im-old", force=True) == "im-new"


def test_snapshot_absent_modal_prunes_only_on_notfound(monkeypatch):
    """Modal sync prunes ONLY on a confirmed NotFound — never on an auth/unknown error (the keep invariant)."""
    import modal
    from modal.exception import AuthError, NotFoundError

    from rllm.sandbox.sandboxed_flow import snapshot_absent

    class _Probe:
        def __init__(self, exc):
            self._exc = exc

        def hydrate(self):
            if self._exc:
                raise self._exc

    monkeypatch.setattr(modal.Image, "from_id", lambda ref: _Probe(NotFoundError("gone")))
    assert snapshot_absent("modal", "im-x") is True  # confirmed gone → prune

    monkeypatch.setattr(modal.Image, "from_id", lambda ref: _Probe(AuthError("nope")))
    assert snapshot_absent("modal", "im-x") is False  # auth → cannot confirm → keep

    monkeypatch.setattr(modal.Image, "from_id", lambda ref: _Probe(None))
    assert snapshot_absent("modal", "im-x") is False  # present → not absent


def test_daytona_ref_absent_reads_enum_or_string_state(monkeypatch):
    """Daytona's snapshot.state is a SnapshotState enum; terminal states prune, others are kept."""
    import daytona

    from rllm.sandbox.backends.daytona import _daytona_ref_absent

    class _Snap:
        def __init__(self, state):
            self.state = state

    def fake_daytona(state):
        svc = type("Svc", (), {"get": lambda self, ref: _Snap(state)})()
        return type("FakeDaytona", (), {"snapshot": svc})()

    enum_state = type("State", (), {"value": "removing"})()  # mimic SnapshotState.REMOVING
    monkeypatch.setattr(daytona, "Daytona", lambda *a, **k: fake_daytona(enum_state))
    assert _daytona_ref_absent("rllm-env-x") is True  # terminal enum → absent

    active_state = type("State", (), {"value": "active"})()
    monkeypatch.setattr(daytona, "Daytona", lambda *a, **k: fake_daytona(active_state))
    assert _daytona_ref_absent("rllm-env-x") is False  # present → keep

    monkeypatch.setattr(daytona, "Daytona", lambda *a, **k: fake_daytona("error"))  # plain string also works
    assert _daytona_ref_absent("rllm-env-x") is True
