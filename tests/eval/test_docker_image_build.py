from __future__ import annotations

import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from rllm.eval._resolution import _build_docker_image


def test_build_docker_image_is_cached_and_locked(monkeypatch, tmp_path):
    context_dir = tmp_path / "environment"
    context_dir.mkdir()
    (context_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("RLLM_DOCKER_BUILD_LOCK_DIR", str(lock_dir))

    built_fingerprint = None
    build_calls = 0
    state_lock = threading.Lock()

    def fake_run(args, **kwargs):
        nonlocal built_fingerprint, build_calls
        if args[:3] == ["docker", "image", "inspect"]:
            with state_lock:
                fingerprint = built_fingerprint
            if fingerprint is None:
                return subprocess.CompletedProcess(args, 1, "", "not found")
            return subprocess.CompletedProcess(args, 0, fingerprint + "\n", "")
        if args[:2] == ["docker", "build"]:
            time.sleep(0.02)
            label = args[args.index("--label") + 1]
            with state_lock:
                build_calls += 1
                built_fingerprint = label.split("=", 1)[1]
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"Unexpected subprocess: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    start = threading.Event()

    def build():
        start.wait()
        return _build_docker_image(context_dir, "shared-task")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(build) for _ in range(8)]
        start.set()
        tags = [future.result() for future in futures]

    assert tags == ["rllm-task-shared-task"] * 8
    assert build_calls == 1


def test_build_docker_image_rebuilds_changed_context(monkeypatch, tmp_path):
    context_dir = tmp_path / "environment"
    context_dir.mkdir()
    dockerfile = context_dir / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:24.04\n")
    monkeypatch.setenv("RLLM_DOCKER_BUILD_LOCK_DIR", str(tmp_path / "locks"))

    built_fingerprint = None
    build_calls = 0

    def fake_run(args, **kwargs):
        nonlocal built_fingerprint, build_calls
        if args[:3] == ["docker", "image", "inspect"]:
            if built_fingerprint is None:
                return subprocess.CompletedProcess(args, 1, "", "not found")
            return subprocess.CompletedProcess(args, 0, built_fingerprint + "\n", "")
        if args[:2] == ["docker", "build"]:
            label = args[args.index("--label") + 1]
            built_fingerprint = label.split("=", 1)[1]
            build_calls += 1
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"Unexpected subprocess: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _build_docker_image(context_dir, "mutable-task")
    dockerfile.write_text("FROM ubuntu:24.04\nRUN echo changed\n")
    _build_docker_image(context_dir, "mutable-task")

    assert build_calls == 2
