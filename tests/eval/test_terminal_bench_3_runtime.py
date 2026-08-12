"""Focused native-runtime coverage for Terminal-Bench 3 contracts."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from rllm.eval._resolution import _create_sandbox_for_task, _create_verifier_sandbox, _sandbox_resource_kwargs, _validate_task_runtime
from rllm.eval.script_evaluator import ShellScriptEvaluator, normalize_artifacts
from rllm.types import Episode, Task, Trajectory


class _Sandbox:
    def __init__(self, *, reward: str | None = None):
        self.reward = reward
        self.execs: list[str] = []
        self.uploaded_files: list[tuple[str, str]] = []
        self.uploaded_dirs: list[tuple[str, str]] = []
        self.closed = False

    def exec(self, command, timeout=None, user=None):  # noqa: ARG002
        self.execs.append(command)
        if command.startswith("test -f /logs/verifier/reward.txt"):
            return "yes" if self.reward is not None else "no"
        if command.startswith("cat /logs/verifier/reward.txt"):
            return self.reward or ""
        if command.startswith("if [ -d "):
            return "directory"
        return ""

    def upload_file(self, local_path, remote_path):
        self.uploaded_files.append((local_path, remote_path))

    def upload_dir(self, local_path, remote_path):
        self.uploaded_dirs.append((local_path, remote_path))

    def download_file(self, remote_path):  # noqa: ARG002
        raise FileNotFoundError(remote_path)

    def close(self):
        self.closed = True


class _ComposeSandbox(_Sandbox):
    """Fake satisfying the full ComposeSandbox protocol (isinstance-checked at runtime)."""

    def __init__(self):
        super().__init__()
        self.events: list[tuple[str, str]] = []

    def is_alive(self) -> bool:
        return not self.closed

    def set_env(self, env):  # noqa: ARG002
        pass

    def service_exec(self, service, command, timeout=None, user=None):  # noqa: ARG002
        self.events.append(("exec", service))
        if command.startswith("if [ -d "):
            return "file"
        return ""

    def service_download_file(self, service, remote_path):
        self.events.append(("download", service))
        return b"sidecar evidence"

    def stop_service(self, service):
        self.events.append(("stop", service))


class _ExecutableArtifactSandbox(_Sandbox):
    def exec(self, command, timeout=None, user=None):  # noqa: ARG002
        if command.startswith("if [ -d /tmp/tool"):
            return "file"
        if command.startswith("stat -c"):
            return "755\n"
        return super().exec(command, timeout=timeout, user=user)

    def download_file(self, remote_path):  # noqa: ARG002
        return b"#!/bin/sh\n"


def _task(tmp_path: Path) -> Task:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/sh\n")
    (tests / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY . /tests\n")
    return Task(
        id="tb3",
        instruction="solve",
        metadata={"environment": {}, "verifier_mode": "separate"},
        dataset_dir=tmp_path,
    )


def _episode() -> Episode:
    return Episode(task="tb3", trajectories=[Trajectory(name="agent", steps=[])])


def test_artifact_declarations_normalize_to_one_shape():
    specs = normalize_artifacts(
        [
            "/tmp/result.json",
            {"source": "/workspace/app", "exclude": ["node_modules"], "service": "main"},
            {"source": "/tmp/db.dump", "service": "postgres"},
        ]
    )
    assert [(spec.source, spec.service, spec.exclude) for spec in specs] == [
        ("/tmp/result.json", "main", ()),
        ("/workspace/app", "main", ("node_modules",)),
        ("/tmp/db.dump", "postgres", ()),
    ]


def test_directory_artifact_is_uploaded_at_original_path(tmp_path, monkeypatch):
    import rllm.eval.script_evaluator as module

    agent = _Sandbox()
    verifier = _Sandbox(reward="1.0")

    def fake_download_directory(sandbox, artifact, target, user=None):  # noqa: ARG001
        path = target / "app"
        path.mkdir(parents=True)
        (path / "solution.py").write_text("pass\n")
        return path

    monkeypatch.setattr(module, "_download_directory", fake_download_directory)
    evaluator = ShellScriptEvaluator(
        sandbox=agent,
        verifier_sandbox_factory=lambda: verifier,
        verifier_tests_baked=True,
        artifacts=[{"source": "/workspace/app", "exclude": ["node_modules"]}],
    )

    output = evaluator.evaluate(_task(tmp_path), _episode())

    assert output.reward == 1.0
    assert verifier.uploaded_dirs[0][1] == "/workspace/app"
    assert "mkdir -p /workspace/app && chmod 777 /workspace/app" in verifier.execs
    assert verifier.closed is True


def test_separate_verifier_does_not_inherit_agent_workdir(tmp_path):
    task = _task(tmp_path)
    task.metadata["workdir"] = "/app"
    agent = _Sandbox()
    verifier = _Sandbox(reward="1.0")
    evaluator = ShellScriptEvaluator(
        sandbox=agent,
        verifier_sandbox_factory=lambda: verifier,
        verifier_tests_baked=True,
    )

    output = evaluator.evaluate(task, _episode())

    assert output.reward == 1.0
    verifier_command = next(command for command in verifier.execs if "/tests/test.sh" in command)
    assert "cd /app" not in verifier_command


def test_regular_artifact_preserves_executable_mode(tmp_path):
    agent = _ExecutableArtifactSandbox()
    verifier = _Sandbox(reward="1.0")
    uploaded_mode = None

    def capture_upload(local_path, remote_path):
        nonlocal uploaded_mode
        uploaded_mode = stat.S_IMODE(Path(local_path).stat().st_mode)
        verifier.uploaded_files.append((local_path, remote_path))

    verifier.upload_file = capture_upload
    evaluator = ShellScriptEvaluator(
        sandbox=agent,
        verifier_sandbox_factory=lambda: verifier,
        verifier_tests_baked=True,
        artifacts=["/tmp/tool"],
    )

    output = evaluator.evaluate(_task(tmp_path), _episode())

    assert output.reward == 1.0
    assert uploaded_mode == 0o755


def test_verifier_build_uses_tests_dockerfile(tmp_path, monkeypatch):
    import rllm.eval._resolution as module

    task = _task(tmp_path)
    built: dict = {}
    sandbox = _Sandbox()

    def fake_build(context, task_id):
        built.update(context=context, task_id=task_id)
        return "verifier:image"

    def fake_create(task, backend, **kwargs):  # noqa: ARG001
        built.update(backend=backend, image=kwargs["image"])
        return sandbox

    monkeypatch.setattr(module, "_build_docker_image", fake_build)
    monkeypatch.setattr(module, "_create_base_sandbox", fake_create)

    assert _create_verifier_sandbox(task, "docker") is sandbox
    assert built["context"] == tmp_path / "tests"
    assert built["image"] == "verifier:image"


def test_sidecar_collection_stops_main_before_collecting(tmp_path):
    agent = _ComposeSandbox()
    verifier = _Sandbox(reward="1.0")
    evaluator = ShellScriptEvaluator(
        sandbox=agent,
        verifier_sandbox_factory=lambda: verifier,
        verifier_tests_baked=True,
        collect_commands=[{"service": "api", "command": "snapshot", "timeout_sec": 10}],
        artifacts=[{"source": "/tmp/evidence.json", "service": "api"}],
    )

    output = evaluator.evaluate(_task(tmp_path), _episode())

    assert output.reward == 1.0
    assert agent.events.index(("stop", "main")) < agent.events.index(("exec", "api"))
    assert ("download", "api") in agent.events
    assert verifier.uploaded_files[0][1] == "/tmp/evidence.json"


def test_compose_task_accepts_modal_but_rejects_unsupported_backend(tmp_path):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "docker-compose.yaml").write_text("services:\n  main:\n    image: python:3.12\n")
    task = Task(id="compose", instruction="", metadata={}, dataset_dir=tmp_path)

    _validate_task_runtime(task, "modal")
    with pytest.raises(RuntimeError, match="requires Docker Compose"):
        _validate_task_runtime(task, "daytona")


def test_compose_task_routes_to_modal_vm_backend(tmp_path, monkeypatch):
    import rllm.sandbox.backends.modal_compose as module

    environment = tmp_path / "environment"
    environment.mkdir()
    compose = environment / "docker-compose.yaml"
    compose.write_text("services:\n  main:\n    image: python:3.12\n")
    task = Task(
        id="compose",
        instruction="",
        metadata={"environment": {"cpus": 2, "memory_mb": 4096, "build_timeout_sec": 75}},
        dataset_dir=tmp_path,
    )
    captured = {}

    class FakeModalCompose:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "ModalComposeSandbox", FakeModalCompose)
    sandbox = _create_sandbox_for_task(task, "modal", name="trial")

    assert isinstance(sandbox, FakeModalCompose)
    assert captured["compose_file"] == compose
    assert captured["build_timeout"] == 75.0
    assert captured["cpu"] == 2.0
    assert captured["memory"] == 4096


def test_gpu_requirements_reach_docker_and_modal():
    task = Task(
        id="gpu",
        instruction="",
        metadata={"environment": {"gpus": 1, "gpu_types": ["H100"]}},
        dataset_dir=Path("."),
    )
    assert _sandbox_resource_kwargs(task, "docker")["gpus"] == 1
    assert _sandbox_resource_kwargs(task, "modal")["gpu"] == "H100"
