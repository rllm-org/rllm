"""Docker Compose-backed sandbox running inside one Modal VM Sandbox."""

from __future__ import annotations

import logging
import shlex
import tempfile
import time
import uuid
from pathlib import Path

import yaml

from rllm.sandbox.backends.compose_sandbox import BaseComposeSandbox
from rllm.sandbox.backends.modal_backend import ModalSandbox

logger = logging.getLogger(__name__)

_REMOTE_ROOT = "/rllm"
_REMOTE_ENVIRONMENT = f"{_REMOTE_ROOT}/environment"
_REMOTE_COMPOSE = f"{_REMOTE_ROOT}/compose"


class ModalComposeSandbox(BaseComposeSandbox):
    """Expose a nested Compose project's ``main`` service as an rLLM Sandbox.

    One Modal VM Sandbox runs Docker-in-Docker; the task's ``main`` container
    and sidecars live in a normal Compose bridge network inside that VM. This
    follows Harbor's Modal DinD topology while retaining rLLM's small Sandbox
    interface; satisfies :class:`rllm.sandbox.protocol.ComposeSandbox`
    structurally.
    """

    def __init__(
        self,
        *,
        name: str,
        environment_dir: Path,
        compose_file: Path,
        resources: dict | None = None,
        build_timeout: float = 600.0,
        dind_image: str = "docker:28.3.3-dind",
        **modal_kwargs,
    ) -> None:
        self.name = name
        self.image = "<modal-compose>"
        self._environment_dir = environment_dir.resolve()
        self._compose_file = compose_file.resolve()
        self._project = self._compose_project_name(name)
        # Invocation-view paths for the shared spec (in-VM paths: compose runs
        # inside the DinD VM, against the staged copy of the project).
        self._compose_env_dir = _REMOTE_ENVIRONMENT
        self._compose_base_file = f"{_REMOTE_COMPOSE}/compose.base.yaml"
        self._compose_task_file = f"{_REMOTE_ENVIRONMENT}/{self._compose_file.name}"
        self._persistent_env: dict[str, str] = {}
        self._closed = False

        if modal_kwargs.get("gpu"):
            raise RuntimeError("Modal VM Sandboxes do not support GPUs; Compose tasks must use CPU resources")

        modal_kwargs.pop("gpu", None)
        self._outer = ModalSandbox(
            name=name,
            image=dind_image,
            entrypoint=None,
            shell="sh",
            experimental_options={"vm_runtime": True},
            block_network=False,
            **modal_kwargs,
        )
        try:
            self._wait_for_docker()
            self._stage_project(resources or {})
            self._run_compose(
                "up",
                "--build",
                "--wait",
                "--wait-timeout",
                str(max(1, int(build_timeout))),
                timeout=build_timeout,
            )
            self.service_exec("main", "true", timeout=30)
        except BaseException:
            self.close()
            raise

    def _wait_for_docker(self) -> None:
        last_error: Exception | None = None
        for _ in range(30):
            try:
                self._outer.exec("docker info >/dev/null", timeout=10)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(2)
        raise RuntimeError(f"Docker daemon did not become ready in Modal sandbox {self.name}: {last_error}")

    def _stage_project(self, resources: dict) -> None:
        self._outer.upload_dir(str(self._environment_dir), _REMOTE_ENVIRONMENT)
        with tempfile.TemporaryDirectory(prefix="rllm-modal-compose-") as tempdir:
            base_file = Path(tempdir) / "compose.base.yaml"
            base_file.write_text(yaml.safe_dump(self._base_overlay(resources), sort_keys=False))
            self._outer.upload_file(str(base_file), self._compose_base_file)

    def _compose_command(self, *args: str) -> str:
        return shlex.join(self._compose_parts(*args))

    def _run_compose(self, *args: str, timeout: float | None = None, check: bool = True) -> str:
        command = self._compose_command(*args)
        try:
            return self._outer.exec(command, timeout=timeout)
        except RuntimeError:
            if check:
                raise
            return ""

    def _host_exec_unchecked(self, command: str) -> None:
        try:
            self._outer.exec(command, timeout=30)
        except RuntimeError:
            pass

    def set_env(self, env: dict[str, str]) -> None:
        self._persistent_env.update({str(k): str(v) for k, v in (env or {}).items()})

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        return self.service_exec("main", command, timeout=timeout, user=user)

    def service_exec(
        self,
        service: str,
        command: str,
        timeout: float | None = None,
        user: str | None = None,
    ) -> str:
        args = ["exec", "-T"]
        if user is not None:
            args.extend(["--user", str(user)])
        for key, value in self._persistent_env.items():
            args.extend(["--env", f"{key}={value}"])
        shell = "bash" if service == "main" else "sh"
        args.extend([service, shell, "-c", command])
        return self._run_compose(*args, timeout=timeout)

    def _stage_path(self) -> str:
        return f"/tmp/.rllm-compose-{uuid.uuid4().hex}"

    def upload_file(self, local_path: str, remote_path: str) -> None:
        staged = self._stage_path()
        try:
            self._outer.upload_file(local_path, staged)
            parent = str(Path(remote_path).parent)
            self.service_exec("main", f"mkdir -p {shlex.quote(parent)}", timeout=30, user="root")
            self._run_compose("cp", staged, f"main:{remote_path}", timeout=300)
        finally:
            self._host_exec_unchecked(f"rm -f {shlex.quote(staged)}")

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        staged = self._stage_path()
        try:
            self._outer.upload_dir(local_path, staged)
            self.service_exec("main", f"mkdir -p {shlex.quote(remote_path)}", timeout=30, user="root")
            self._run_compose("cp", f"{staged}/.", f"main:{remote_path}", timeout=600)
        finally:
            self._host_exec_unchecked(f"rm -rf {shlex.quote(staged)}")

    def download_file(self, remote_path: str) -> bytes:
        return self.service_download_file("main", remote_path)

    def service_download_file(self, service: str, remote_path: str) -> bytes:
        staged = self._stage_path()
        try:
            self._run_compose("cp", f"{service}:{remote_path}", staged, timeout=300)
            return self._outer.download_file(staged)
        finally:
            self._host_exec_unchecked(f"rm -f {shlex.quote(staged)}")

    def stop_service(self, service: str) -> None:
        self._run_compose("stop", "--timeout", "10", service, timeout=30)

    def is_alive(self) -> bool:
        if not self._outer.is_alive():
            return False
        try:
            self.service_exec("main", "true", timeout=10)
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._run_compose("down", "--volumes", "--remove-orphans", "--timeout", "10", timeout=60, check=False)
        finally:
            self._outer.close()
        logger.info("ModalComposeSandbox %s closed", self.name)
