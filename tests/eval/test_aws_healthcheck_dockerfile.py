"""Offline tests for the harbor-format additions: the readiness healthcheck step, the
``from_dockerfile`` build selection on remote backends, and COPY-aware snapshot identity.

All hermetic — no docker/daytona/network. Fixtures build a task dir with an
``environment/Dockerfile`` (+ optional context files) and a fake sandbox that records execs.
"""

from pathlib import Path

import pytest

from rllm.eval._resolution import _builds_from_dockerfile, _run_healthcheck, _task_dockerfile
from rllm.sandbox.snapshot import env_key_for
from rllm.types import Task


class _Sb:
    """Records exec() calls; optionally fails to simulate an unready service."""

    def __init__(self, fail: bool = False):
        self.calls: list[str] = []
        self.fail = fail

    def exec(self, command: str, timeout=None, user=None) -> str:
        self.calls.append(command)
        if self.fail:
            raise RuntimeError("not ready")
        return ""


def _task(tmp: Path, *, dockerfile: str | None = "FROM x\n", files: dict | None = None, backend: str = "daytona", replay=None, healthcheck=None) -> Task:
    env = tmp / "environment"
    env.mkdir(parents=True, exist_ok=True)
    if dockerfile is not None:
        (env / "Dockerfile").write_text(dockerfile)
    for rel, content in (files or {}).items():
        p = env / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    environment: dict = {}
    if replay is not None:
        environment["replay_dockerfile"] = replay
    if healthcheck is not None:
        environment["healthcheck"] = healthcheck
    return Task(id="t", instruction="", metadata={"environment": environment, "sandbox_backend": backend}, dataset_dir=tmp)


# ── healthcheck step ──────────────────────────────────────────────────────────


def test_healthcheck_noop_without_declaration(tmp_path):
    sb = _Sb()
    _run_healthcheck(_task(tmp_path), sb)  # no [environment.healthcheck]
    assert sb.calls == []  # strict no-op — protects non-service eval/training tasks


def test_healthcheck_runs_command(tmp_path):
    sb = _Sb()
    task = _task(tmp_path, healthcheck={"command": "bash /usr/local/bin/start_localstack.sh", "timeout_sec": 30})
    _run_healthcheck(task, sb)
    assert sb.calls == ["bash /usr/local/bin/start_localstack.sh"]


def test_healthcheck_raises_after_retries(tmp_path):
    sb = _Sb(fail=True)
    task = _task(tmp_path, healthcheck={"command": "false", "retries": 2, "interval_sec": 0, "timeout_sec": 1})
    with pytest.raises(RuntimeError, match="Healthcheck failed"):
        _run_healthcheck(task, sb)
    assert len(sb.calls) == 3  # retries=2 → 3 attempts


# ── from_dockerfile build selection ────────────────────────────────────────────


def test_builds_from_dockerfile_gating(tmp_path):
    # daytona + Dockerfile + replay(default True) → build the real Dockerfile
    t = _task(tmp_path / "a", backend="daytona")
    assert _builds_from_dockerfile(t, "daytona") == _task_dockerfile(t)
    # docker builds itself → never the from_dockerfile path here
    assert _builds_from_dockerfile(t, "docker") is None
    # prebuilt-image task (replay_dockerfile = false) boots as-is, no rebuild
    assert _builds_from_dockerfile(_task(tmp_path / "b", replay=False), "daytona") is None
    # no Dockerfile on disk → nothing to build
    assert _builds_from_dockerfile(_task(tmp_path / "c", dockerfile=None), "daytona") is None


# ── COPY-aware snapshot identity ───────────────────────────────────────────────


def test_env_key_distinguishes_copy_context(tmp_path):
    df = "FROM localstack/localstack:4.4.0\nCOPY init/ready.d/ /etc/localstack/init/ready.d/\n"
    a = _task(tmp_path / "a", dockerfile=df, files={"init/ready.d/seed.py": "TABLE_A"})
    b = _task(tmp_path / "b", dockerfile=df, files={"init/ready.d/seed.py": "TABLE_B"})
    a2 = _task(tmp_path / "a2", dockerfile=df, files={"init/ready.d/seed.py": "TABLE_A"})
    # Same FROM+RUN block, different COPYed seed data → distinct snapshot/warm-queue keys.
    assert env_key_for(a, "daytona") != env_key_for(b, "daytona")
    # Identical context → identical key (GRPO copies / re-runs share one snapshot).
    assert env_key_for(a, "daytona") == env_key_for(a2, "daytona")
