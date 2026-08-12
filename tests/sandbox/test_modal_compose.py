"""Unit tests for Docker Compose running inside one Modal VM Sandbox."""

from __future__ import annotations

from pathlib import Path

import yaml

from rllm.sandbox.backends.modal_compose import ModalComposeSandbox


class _OuterSandbox:
    instances: list[_OuterSandbox] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.commands: list[tuple[str, float | None]] = []
        self.uploaded_dirs: list[tuple[str, str]] = []
        self.uploaded_files: list[tuple[str, str]] = []
        self.base_compose: dict | None = None
        self.closed = False
        self.alive = True
        self.__class__.instances.append(self)

    def exec(self, command: str, timeout: float | None = None, user=None):  # noqa: ARG002
        self.commands.append((command, timeout))
        return ""

    def upload_dir(self, local_path: str, remote_path: str):
        self.uploaded_dirs.append((local_path, remote_path))

    def upload_file(self, local_path: str, remote_path: str):
        self.uploaded_files.append((local_path, remote_path))
        if remote_path.endswith("compose.base.yaml"):
            self.base_compose = yaml.safe_load(Path(local_path).read_text())

    def download_file(self, remote_path: str):
        return f"downloaded:{remote_path}".encode()

    def is_alive(self):
        return self.alive

    def close(self):
        self.closed = True


def _environment(tmp_path: Path) -> tuple[Path, Path]:
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n")
    compose = environment / "docker-compose.yaml"
    compose.write_text("services:\n  main: {}\n  api:\n    image: nginx\n")
    return environment, compose


def _sandbox(tmp_path: Path, monkeypatch) -> tuple[ModalComposeSandbox, _OuterSandbox]:
    import rllm.sandbox.backends.modal_compose as module

    _OuterSandbox.instances.clear()
    monkeypatch.setattr(module, "ModalSandbox", _OuterSandbox)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    environment, compose = _environment(tmp_path)
    sandbox = ModalComposeSandbox(
        name="TB3 / Modal Trial",
        environment_dir=environment,
        compose_file=compose,
        resources={"cpus": 2, "memory_mb": 4096},
        build_timeout=90,
        cpu=2,
        memory=4096,
        timeout=1200,
    )
    return sandbox, _OuterSandbox.instances[-1]


def test_modal_compose_starts_dind_vm_and_remote_project(tmp_path, monkeypatch):
    sandbox, outer = _sandbox(tmp_path, monkeypatch)

    assert outer.kwargs["image"] == "docker:28.3.3-dind"
    assert outer.kwargs["entrypoint"] is None
    assert outer.kwargs["shell"] == "sh"
    assert outer.kwargs["experimental_options"] == {"vm_runtime": True}
    assert outer.kwargs["block_network"] is False
    assert outer.kwargs["cpu"] == 2
    assert outer.kwargs["memory"] == 4096
    assert outer.uploaded_dirs[0][1] == "/rllm/environment"
    assert outer.base_compose["services"]["main"]["build"]["context"] == "/rllm/environment"
    assert outer.base_compose["services"]["main"]["cpus"] == 2.0
    assert any("docker compose" in command and "up --build --wait" in command for command, _ in outer.commands)
    sandbox.close()


def test_exec_targets_main_and_sidecar_with_user_and_env(tmp_path, monkeypatch):
    sandbox, outer = _sandbox(tmp_path, monkeypatch)
    sandbox.set_env({"API_KEY": "secret value"})

    sandbox.exec("echo main", timeout=12, user="agent")
    sandbox.service_exec("api", "echo sidecar", timeout=13, user="root")

    main = next(command for command, _ in outer.commands if "echo main" in command)
    sidecar = next(command for command, _ in outer.commands if "echo sidecar" in command)
    assert "exec -T --user agent --env 'API_KEY=secret value' main bash -c 'echo main'" in main
    assert "exec -T --user root --env 'API_KEY=secret value' api sh -c 'echo sidecar'" in sidecar
    sandbox.close()


def test_file_transfers_bridge_outer_vm_and_nested_services(tmp_path, monkeypatch):
    sandbox, outer = _sandbox(tmp_path, monkeypatch)
    local_file = tmp_path / "input.txt"
    local_file.write_text("hello")
    local_dir = tmp_path / "tests"
    local_dir.mkdir()
    (local_dir / "test.sh").write_text("true\n")

    sandbox.upload_file(str(local_file), "/workspace/input.txt")
    sandbox.upload_dir(str(local_dir), "/tests")
    result = sandbox.service_download_file("api", "/tmp/evidence.json")

    assert result.startswith(b"downloaded:/tmp/.rllm-compose-")
    commands = [command for command, _ in outer.commands]
    assert any("cp /tmp/.rllm-compose-" in command and "main:/workspace/input.txt" in command for command in commands)
    assert any("main:/tests" in command for command in commands)
    assert any("api:/tmp/evidence.json" in command for command in commands)
    sandbox.close()


def test_close_stops_compose_and_modal_vm_once(tmp_path, monkeypatch):
    sandbox, outer = _sandbox(tmp_path, monkeypatch)

    sandbox.stop_service("main")
    sandbox.close()
    sandbox.close()

    commands = [command for command, _ in outer.commands]
    assert any("stop --timeout 10 main" in command for command in commands)
    assert sum("down --volumes --remove-orphans" in command for command in commands) == 1
    assert outer.closed is True


def test_is_alive_requires_outer_vm_and_main_service(tmp_path, monkeypatch):
    sandbox, outer = _sandbox(tmp_path, monkeypatch)
    assert sandbox.is_alive() is True
    outer.alive = False
    assert sandbox.is_alive() is False
    sandbox.close()
