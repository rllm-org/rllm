"""Docker Compose-backed sandbox for multi-service benchmark tasks."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

from rllm.sandbox.backends.docker import DockerSandbox

logger = logging.getLogger(__name__)


class DockerComposeSandbox:
    """Expose a Compose project's ``main`` service through the Sandbox API.

    Task-authored Compose files are overlays: rLLM supplies the base ``main``
    build and keepalive command, while the task adds sidecars, networking,
    healthchecks, volumes, and any main-service overrides.
    """

    def __init__(
        self,
        *,
        name: str,
        environment_dir: Path,
        compose_file: Path,
        resources: dict | None = None,
        build_timeout: float = 600.0,
    ) -> None:
        import docker

        self.name = name
        self.image = "<docker-compose>"
        self._environment_dir = environment_dir.resolve()
        self._compose_file = compose_file.resolve()
        self._client = docker.from_env()
        self._handles: dict[str, DockerSandbox] = {}
        self._closed = False
        self._tempdir = tempfile.TemporaryDirectory(prefix="rllm-compose-")
        self._base_file = Path(self._tempdir.name) / "compose.base.yaml"
        self._project = _compose_project_name(name)
        self._write_base(resources or {})
        try:
            self._run_compose(
                "up",
                "--build",
                "--wait",
                "--wait-timeout",
                str(max(1, int(build_timeout))),
                timeout=build_timeout,
            )
            self._handle("main")
        except BaseException:
            self.close()
            raise

    def _write_base(self, resources: dict) -> None:
        main: dict = {
            "build": {"context": str(self._environment_dir), "dockerfile": "Dockerfile"},
            "command": ["sleep", "infinity"],
        }
        cpus = resources.get("cpus")
        memory_mb = resources.get("memory_mb")
        env = resources.get("env")
        if cpus:
            main["cpus"] = float(cpus)
        if memory_mb:
            main["mem_limit"] = f"{int(memory_mb)}m"
        if env:
            main["environment"] = {str(k): str(v) for k, v in env.items()}
        self._base_file.write_text(yaml.safe_dump({"services": {"main": main}}, sort_keys=False))

    def _command(self, *args: str) -> list[str]:
        return [
            "docker",
            "compose",
            "--project-name",
            self._project,
            "--project-directory",
            str(self._environment_dir),
            "-f",
            str(self._base_file),
            "-f",
            str(self._compose_file),
            *args,
        ]

    def _run_compose(self, *args: str, timeout: float | None = None, check: bool = True) -> str:
        result = subprocess.run(
            self._command(*args),
            cwd=str(self._environment_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout)[-4000:]
            raise RuntimeError(f"docker compose {' '.join(args)} failed for {self.name}:\n{detail}")
        return result.stdout

    def _handle(self, service: str) -> DockerSandbox:
        cached = self._handles.get(service)
        if cached is not None:
            return cached
        container_id = self._run_compose("ps", "-q", service).strip()
        if not container_id:
            raise RuntimeError(f"Docker Compose service {service!r} is not running for {self.name}")
        container = self._client.containers.get(container_id)
        handle = DockerSandbox.from_container(f"{self.name}:{service}", container, self._client)
        self._handles[service] = handle
        return handle

    def set_env(self, env: dict[str, str]) -> None:
        self._handle("main").set_env(env)

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        return self.service_exec("main", command, timeout=timeout, user=user)

    def service_exec(
        self,
        service: str,
        command: str,
        timeout: float | None = None,
        user: str | None = None,
    ) -> str:
        handle = self._handle(service)
        if service == "main":
            return handle.exec(command, timeout=timeout, user=user)
        return handle.exec_with_shell(command, shell="sh", timeout=timeout, user=user)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self._handle("main").upload_file(local_path, remote_path)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        self._handle("main").upload_dir(local_path, remote_path)

    def download_file(self, remote_path: str) -> bytes:
        return self.service_download_file("main", remote_path)

    def service_download_file(self, service: str, remote_path: str) -> bytes:
        return self._handle(service).download_file(remote_path)

    def stop_service(self, service: str) -> None:
        self._run_compose("stop", "--timeout", "10", service, timeout=30)

    def is_alive(self) -> bool:
        try:
            return self._handle("main").is_alive()
        except Exception:
            return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._run_compose("down", "--volumes", "--remove-orphans", "--timeout", "10", timeout=60, check=False)
        finally:
            self._handles.clear()
            self._tempdir.cleanup()
        logger.info("DockerComposeSandbox %s closed", self.name)


def _compose_project_name(name: str) -> str:
    """Return an isolated, Compose-safe project name."""
    normalized = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-_")
    return (normalized or "rllm")[:63]
