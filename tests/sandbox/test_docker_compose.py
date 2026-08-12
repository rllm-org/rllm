"""Unit tests for the native Docker Compose sandbox."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import yaml

from rllm.sandbox.backends.docker_compose import DockerComposeSandbox, _compose_project_name


class _Container:
    def __init__(self):
        self.image = SimpleNamespace(tags=["task:image"])
        self.status = "running"

    def reload(self):
        return None


class _Containers:
    def __init__(self):
        self.items = {"cid-main": _Container(), "cid-api": _Container()}

    def get(self, container_id):
        return self.items[container_id]


def test_compose_sandbox_layers_task_file_and_uses_unique_project(tmp_path, monkeypatch):
    import rllm.sandbox.backends.docker_compose as module

    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n")
    overlay = environment / "docker-compose.yaml"
    overlay.write_text("services:\n  main:\n    depends_on:\n      api:\n        condition: service_started\n  api:\n    image: nginx\n")

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        calls.append(command)
        if "ps" in command:
            service = command[-1]
            return SimpleNamespace(returncode=0, stdout=f"cid-{service}\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    client = SimpleNamespace(containers=_Containers())
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "docker", SimpleNamespace(from_env=lambda: client))

    sandbox = DockerComposeSandbox(
        name="TB3/Trial 1",
        environment_dir=environment,
        compose_file=overlay,
        resources={"cpus": 2, "memory_mb": 4096},
    )
    base = yaml.safe_load(sandbox._base_file.read_text())
    assert base["services"]["main"]["build"]["context"] == str(environment.resolve())
    assert base["services"]["main"]["cpus"] == 2.0
    assert any("up" in call and "--wait" in call for call in calls)
    assert sandbox._handle("api").name.endswith(":api")
    sandbox.stop_service("main")
    sandbox.close()
    assert any("stop" in call for call in calls)
    assert any("down" in call and "--volumes" in call for call in calls)


def test_compose_project_name_is_safe_and_bounded():
    name = _compose_project_name("TB3 / Weird Task ! " + "x" * 100)
    assert name.startswith("tb3-weird-task-")
    assert len(name) <= 63
