"""Live checks for the boxd sandbox backend against real boxd machines.

Skipped unless ``BOXD_E2E=1`` and ``BOXD_API_KEY`` are set. Each test
creates its own machines and deletes them through ``close()``::

    BOXD_E2E=1 BOXD_API_KEY=bxd_... BOXD_REGION=fra \\
        pytest tests/sandbox/test_boxd_live.py -v -s

``RLLM_BOXD_NAME_PREFIX`` and ``RLLM_BOXD_LABELS`` mark the machines a
run creates.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BOXD_E2E") != "1" or not os.environ.get("BOXD_API_KEY"),
    reason="live boxd test: set BOXD_E2E=1 and BOXD_API_KEY",
)

IMAGE = os.environ.get("BOXD_E2E_IMAGE", "python:3.11-slim")


def _name(tag: str) -> str:
    return f"e2e-{tag}-{uuid.uuid4().hex[:6]}"


def _report(sb, what: str, t0: float) -> None:
    print(f"\n[boxd-e2e] {what}: machine {sb.machine_id} ({sb.machine_name}) {time.monotonic() - t0:.2f}s")


@pytest.fixture
def sandbox():
    from rllm.sandbox.backends.boxd import BoxdSandbox

    t0 = time.monotonic()
    sb = BoxdSandbox(name=_name("sb"), image=IMAGE, env={"RLLM_E2E": "from-create"})
    _report(sb, "create+ready", t0)
    try:
        yield sb
    finally:
        sb.close()


def test_lifecycle_exec_files_and_delete(sandbox, tmp_path):
    import boxd

    sb = sandbox
    assert sb.is_alive()

    t0 = time.monotonic()
    assert sb.exec("echo hello") == "hello\n"
    print(f"[boxd-e2e] first exec {time.monotonic() - t0:.2f}s")

    # Exit code, stdout and stderr stay apart.
    with pytest.raises(RuntimeError, match=r"exit 3[\s\S]*to-stderr"):
        sb.exec("echo to-stdout; echo to-stderr >&2; exit 3")
    assert sb.exec("echo out; echo err >&2") == "out\n"

    # The task image is what runs, and env given at create reaches exec.
    assert "3.11" in sb.exec("python3 -V")
    assert sb.exec("echo $RLLM_E2E").strip() == "from-create"

    # Shell features through bash -c.
    assert sb.exec("for i in 1 2 3; do printf $i; done; [[ -n x ]] && echo") == "123\n"

    # Timeout ends the call and the remote process.
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        sb.exec("sleep 30; touch /tmp/should-not-exist", timeout=3)
    assert time.monotonic() - t0 < 20
    time.sleep(1)
    assert sb.exec("test -e /tmp/should-not-exist && echo yes || echo no").strip() == "no"

    # 5 MiB binary round trip.
    blob = os.urandom(5 * 1024 * 1024)
    local = tmp_path / "blob.bin"
    local.write_bytes(blob)
    sb.upload_file(str(local), "/app/data/blob.bin")
    assert sb.exec("sha256sum /app/data/blob.bin").split()[0] == hashlib.sha256(blob).hexdigest()

    # Directory tree with an executable.
    tree = tmp_path / "tests"
    (tree / "sub").mkdir(parents=True)
    (tree / "test.sh").write_text("#!/bin/bash\necho graded\n")
    (tree / "test.sh").chmod(0o755)
    (tree / "sub" / "x.txt").write_text("x")
    sb.upload_dir(str(tree), "/tests")
    assert sb.exec("/tests/test.sh") == "graded\n"
    assert sb.exec("cat /tests/sub/x.txt") == "x"
    assert sb.exec("ls /tmp | grep -c _upload_ || true").strip() == "0"

    # A server and its client inside one exec, the way a harness runs its agent.
    out = sb.exec("python3 -m http.server 8123 >/tmp/http.log 2>&1 & sleep 2; python3 -c \"import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8123/').read().decode())\"; kill %1")
    assert "Directory listing" in out

    # user= switches user; root is a no-op on an image that runs as root.
    assert sb.exec("id -un", user="root").strip() == "root"
    sb.exec("useradd -m agent 2>/dev/null || adduser --disabled-password --gecos '' agent")
    assert sb.exec("id -un", user="agent").strip() == "agent"
    assert sb.exec("id -un", user="1000").strip() == "agent"

    machine_id = sb.machine_id
    sb.close()
    assert not sb.is_alive()
    client = boxd.Boxd(api_key=os.environ["BOXD_API_KEY"], region=os.environ.get("BOXD_REGION") or None)
    with pytest.raises(boxd.errors.NotFoundError):
        client.machines.get(machine_id)
    print(f"[boxd-e2e] deleted: machine {machine_id}")


def test_hibernated_sandbox_is_alive_and_wakes_on_exec(sandbox):
    import boxd

    sb = sandbox
    sb.exec("echo kept > /root/state")
    client = boxd.Boxd(api_key=os.environ["BOXD_API_KEY"], region=os.environ.get("BOXD_REGION") or None)
    client.machines.hibernate(sb.machine_id)
    deadline = time.monotonic() + 120
    while str(client.machines.get(sb.machine_id).status) != "hibernated" and time.monotonic() < deadline:
        time.sleep(1)
    assert str(client.machines.get(sb.machine_id).status) == "hibernated"
    assert sb.is_alive()
    t0 = time.monotonic()
    assert sb.exec("cat /root/state") == "kept\n"
    print(f"[boxd-e2e] wake on exec {time.monotonic() - t0:.2f}s")


def test_exec_on_a_stopped_sandbox_starts_it(sandbox):
    import boxd

    sb = sandbox
    sb.exec("echo kept > /root/state")
    client = boxd.Boxd(api_key=os.environ["BOXD_API_KEY"], region=os.environ.get("BOXD_REGION") or None)
    client.machines.stop(sb.machine_id)
    deadline = time.monotonic() + 120
    while str(client.machines.get(sb.machine_id).status) != "stopped" and time.monotonic() < deadline:
        time.sleep(1)
    assert str(client.machines.get(sb.machine_id).status) == "stopped"
    assert not sb.is_alive()
    t0 = time.monotonic()
    assert sb.exec("cat /root/state") == "kept\n"
    print(f"[boxd-e2e] exec on stopped machine (start + exec) {time.monotonic() - t0:.2f}s")
    assert sb.is_alive()


def test_snapshot_build_restore_and_delete(tmp_path):
    from rllm.sandbox.backends.boxd import BoxdSandbox, _boxd_ref_absent, build_boxd_snapshot, delete_boxd_snapshot
    from rllm.sandbox.protocol import SnapshotNotFound
    from rllm.types import Task

    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(f"FROM {IMAGE}\nRUN pip install -q six && \\\n    echo baked > /opt/marker\n")
    task = Task(id="e2e", instruction="", metadata={"environment": {"docker_image": IMAGE}}, dataset_dir=tmp_path)

    key = f"{os.environ.get('RLLM_BOXD_NAME_PREFIX', '')}rllm-env-e2e-{uuid.uuid4().hex[:8]}"
    t0 = time.monotonic()
    assert build_boxd_snapshot(task, key, install_script="echo installed > /opt/install") == key
    print(f"\n[boxd-e2e] snapshot {key} built in {time.monotonic() - t0:.2f}s")
    assert build_boxd_snapshot(task, key) == key  # reuse, no rebuild
    assert _boxd_ref_absent(key) is False

    t0 = time.monotonic()
    sb = BoxdSandbox(name=_name("restore"), image=key)
    _report(sb, "restore from snapshot", t0)
    try:
        assert sb.exec("cat /opt/marker /opt/install; python3 -c 'import six'") == "baked\ninstalled\n"
    finally:
        sb.close()

    assert delete_boxd_snapshot(key) is True
    assert _boxd_ref_absent(key) is True
    with pytest.raises(SnapshotNotFound):
        BoxdSandbox(name=_name("gone"), image=key)


def test_wrong_api_key_is_a_clear_error():
    import boxd

    from rllm.sandbox.backends.boxd import BoxdSandbox

    with pytest.raises(boxd.errors.AuthenticationError):
        BoxdSandbox(name=_name("badkey"), image=IMAGE, api_key="bxd_" + "0" * 40)


def test_sandbox_on_an_ubuntu_image(tmp_path):
    """``ubuntu:24.04`` boots too, through the same exec and upload path."""
    from rllm.sandbox.backends.boxd import BoxdSandbox

    t0 = time.monotonic()
    sb = BoxdSandbox(name=_name("ubuntu"), image="ubuntu:24.04")
    _report(sb, "create+ready (ubuntu:24.04)", t0)
    try:
        assert "Ubuntu" in sb.exec("cat /etc/os-release")
        assert sb.exec("echo $RLLM_E2E_UNSET; id -un", user="root").strip() == "root"
        f = Path(tmp_path / "x.txt")
        f.write_text("ok")
        sb.upload_file(str(f), "/root/x.txt")
        assert sb.exec("cat /root/x.txt") == "ok"
    finally:
        sb.close()


@pytest.mark.skipif(os.environ.get("BOXD_E2E_LONG") != "1", reason="31-minute check: set BOXD_E2E_LONG=1")
def test_a_command_silent_for_31_minutes_completes(sandbox):
    t0 = time.monotonic()
    assert sandbox.exec("sleep 1860; echo finished", timeout=2400) == "finished\n"
    print(f"[boxd-e2e] silent command finished after {time.monotonic() - t0:.0f}s")
