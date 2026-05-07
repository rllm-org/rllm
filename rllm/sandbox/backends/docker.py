"""Docker container sandbox backend."""

from __future__ import annotations

import io
import logging
import os
import tarfile
import time

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Sandbox implementation using Docker containers.

    Creates a container with ``sleep infinity``, uploads files via tar archives,
    executes commands via ``exec_run()``, and runs agent processes with ``nohup``.

    Requires the ``docker`` Python package (not in ``[sdk]`` extra — install
    separately when using ``backend=docker``).
    """

    def __init__(self, name: str, image: str = "python:3.11-slim", **kwargs):
        import docker

        self.name = name
        self.image = image
        self._client = docker.from_env()
        self._container = self._client.containers.run(
            image,
            command="sleep infinity",
            name=f"rllm-sandbox-{name}",
            detach=True,
            remove=False,
        )
        logger.info("DockerSandbox %s created (container: %s, image: %s)", name, self._container.short_id, image)

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        """Execute a command inside the container.

        Args:
            command: Shell command to run.
            timeout: Optional per-call timeout (currently unused by Docker SDK).
            user: Optional UID/username to run as (e.g., ``"agent"``, ``"1000"``).
                Maps to ``docker exec --user``. If ``None``, runs as the
                container's default user.
        """
        kwargs: dict = {"demux": True}
        if user is not None:
            kwargs["user"] = user
        exit_code, output = self._container.exec_run(
            ["bash", "-c", command],
            **kwargs,
        )
        stdout = (output[0] or b"").decode("utf-8", errors="replace")
        stderr = (output[1] or b"").decode("utf-8", errors="replace")
        if exit_code != 0:
            # Surface enough context to debug verifier failures inside the
            # container — the full pytest/setup log can be megabytes; show
            # the last ~8KB of each stream which is where exit-causing
            # errors live.
            tail_chars = 8000
            stderr_tail = stderr[-tail_chars:] if len(stderr) > tail_chars else stderr
            stdout_tail = stdout[-tail_chars:] if len(stdout) > tail_chars else stdout
            logger.warning(
                "Command failed (exit %d) in container %s: %s\nstdout (tail):\n%s\nstderr (tail):\n%s",
                exit_code,
                self.name,
                command,
                stdout_tail,
                stderr_tail,
            )
            raise RuntimeError(f"Command failed (exit {exit_code}) in container {self.name}: {command}\nstderr (tail):\n{stderr_tail}\nstdout (tail):\n{stdout_tail}")
        return stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file into the container via tar archive."""
        remote_dir = os.path.dirname(remote_path)
        remote_name = os.path.basename(remote_path)
        self._container.exec_run(["mkdir", "-p", remote_dir])

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(local_path, arcname=remote_name)
        tar_stream.seek(0)
        self._container.put_archive(remote_dir, tar_stream)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree into the container via tar archive."""
        remote_parent = os.path.dirname(remote_path.rstrip("/"))
        remote_name = os.path.basename(remote_path.rstrip("/"))
        self._container.exec_run(["mkdir", "-p", remote_parent])

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(local_path, arcname=remote_name)
        tar_stream.seek(0)
        self._container.put_archive(remote_parent, tar_stream)

    def start_agent_process(self, command: str, port: int) -> None:
        """Start a background process in the container using nohup."""
        bg_command = f"nohup {command} > /tmp/worker.log 2>&1 &"
        self._container.exec_run(["bash", "-c", bg_command], detach=True)

        # Wait for the server to become ready
        self._wait_for_ready(port, timeout=30.0)
        logger.info("Agent process started in container %s on port %d", self.name, port)

    def _wait_for_ready(self, port: int, timeout: float = 30.0) -> None:
        """Poll the health endpoint inside the container."""
        start = time.time()
        while time.time() - start < timeout:
            exit_code, _ = self._container.exec_run(
                ["bash", "-c", f"curl -s http://127.0.0.1:{port}/health"],
            )
            if exit_code == 0:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Worker server did not start within {timeout}s in container {self.name}")

    def get_endpoint(self, port: int) -> tuple[str, dict[str, str]]:
        """Return the URL to reach the given port.

        For Docker containers, we use the container's internal IP.
        """
        self._container.reload()
        networks = self._container.attrs["NetworkSettings"]["Networks"]
        # Use the first available network's IP
        for net_info in networks.values():
            ip = net_info.get("IPAddress")
            if ip:
                return f"http://{ip}:{port}", {}
        # Fallback to localhost (works if port is published)
        return f"http://127.0.0.1:{port}", {}

    def close(self) -> None:
        """Stop and remove the container."""
        try:
            self._container.stop(timeout=5)
        except Exception:
            try:
                self._container.kill()
            except Exception:
                pass
        try:
            self._container.remove(force=True)
        except Exception:
            pass
        logger.info("DockerSandbox %s closed", self.name)


def create_docker_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> DockerSandbox:
    """Factory function for creating a DockerSandbox."""
    return DockerSandbox(name=name, image=image, **kwargs)
