"""Docker container sandbox backend."""

from __future__ import annotations

import io
import logging
import os
import tarfile

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
            # The raised message stays short — callers print it at WARNING
            # for *every* failed verifier, and dumping kilobytes of pytest
            # output for each agent that didn't solve a task spams the
            # terminal. The full tail goes to ``logger.debug`` so it's
            # available with ``--log-level=debug`` without polluting the
            # default run.
            short_tail = 600
            err_tail = stderr[-short_tail:] if len(stderr) > short_tail else stderr
            full_tail = 8000
            logger.debug(
                "Command failed (exit %d) in container %s: %s\nstdout (tail):\n%s\nstderr (tail):\n%s",
                exit_code,
                self.name,
                command,
                stdout[-full_tail:] if len(stdout) > full_tail else stdout,
                stderr[-full_tail:] if len(stderr) > full_tail else stderr,
            )
            raise RuntimeError(f"Command failed (exit {exit_code}) in container {self.name}: {command}\nstderr (tail):\n{err_tail}")
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
