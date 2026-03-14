"""Modal sandbox backend.

Uses Modal Sandboxes (https://modal.com/docs/guide/sandboxes) to run
agent code in serverless cloud containers.

Requires the ``modal`` package::

    pip install modal

Authentication is handled via ``modal setup`` or the ``MODAL_TOKEN_ID``
and ``MODAL_TOKEN_SECRET`` environment variables.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
import time

logger = logging.getLogger(__name__)

# Default sandbox timeout: 30 minutes (Modal default is 5 min).
_DEFAULT_TIMEOUT = 30 * 60


class ModalSandbox:
    """Sandbox implementation using Modal serverless containers.

    Creates a Modal Sandbox via ``modal.Sandbox.create()``, executes
    commands via ``sandbox.exec()``, and manages file operations through
    exec-based helpers (Modal's file API is alpha).

    The ``image`` parameter accepts either:
    - A Docker Hub image name (e.g. ``"python:3.11-slim"``) — will be
      wrapped with ``modal.Image.from_registry()``.
    - A ``modal.Image`` object directly.

    Optional kwargs:
    - ``app_name``: Modal App name (default: ``"rllm-sandbox"``).
    - ``timeout``: Sandbox lifetime in seconds (default: 1800).
    - ``secrets``: List of ``modal.Secret`` objects.
    - ``volumes``: Dict mapping mount paths to ``modal.Volume`` objects.
    - ``workdir``: Working directory inside the container.
    - ``gpu``: GPU spec (e.g. ``"T4"``, ``"A10G"``).
    - ``cpu``: CPU count (float).
    - ``memory``: Memory in MB (int).
    """

    def __init__(self, name: str, image: str = "python:3.11-slim", **kwargs):
        import modal

        self.name = name
        self._image_spec = image
        self._timeout = kwargs.pop("timeout", _DEFAULT_TIMEOUT)
        self._app_name = kwargs.pop("app_name", "rllm-sandbox")

        # Resolve image
        if isinstance(image, str):
            modal_image = modal.Image.from_registry(image)
        else:
            # Assume it's already a modal.Image
            modal_image = image

        # Resolve app
        self._app = modal.App.lookup(self._app_name, create_if_missing=True)

        # Build create kwargs
        create_kwargs: dict = {
            "app": self._app,
            "image": modal_image,
            "timeout": self._timeout,
        }

        # Optional params
        for key in ("secrets", "volumes", "workdir", "gpu", "cpu", "memory"):
            if key in kwargs:
                create_kwargs[key] = kwargs.pop(key)

        self._sandbox = modal.Sandbox.create(**create_kwargs)
        self._sandbox_id = self._sandbox.object_id

        logger.info(
            "ModalSandbox %s created (id: %s, image: %s)",
            name,
            self._sandbox_id,
            image if isinstance(image, str) else "<modal.Image>",
        )

    def exec(self, command: str, timeout: float | None = None) -> str:
        """Execute a command inside the Modal sandbox.

        Runs ``bash -c <command>`` and returns stdout. Raises
        ``RuntimeError`` on non-zero exit code (matching DockerSandbox
        behavior).
        """
        exec_kwargs = {}
        if timeout is not None:
            exec_kwargs["timeout"] = int(timeout)

        process = self._sandbox.exec("bash", "-c", command, **exec_kwargs)

        stdout = process.stdout.read()
        stderr = process.stderr.read()

        # Wait for the process to complete and get exit code
        process.wait()
        exit_code = process.returncode

        if exit_code != 0:
            logger.warning(
                "Command failed in sandbox %s: %s\nstderr: %s",
                self.name,
                command,
                stderr[:500],
            )
            raise RuntimeError(f"Command failed (exit {exit_code}) in sandbox {self.name}: {command}\n{stderr[:500]}")
        return stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file into the Modal sandbox.

        Uses ``cat`` via exec to write file contents since Modal's file
        API is in alpha.
        """
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            self._exec_unchecked(f"mkdir -p {remote_dir}")

        with open(local_path, "rb") as f:
            content = f.read()

        # Use base64 encoding for safe binary transfer
        import base64

        b64 = base64.b64encode(content).decode("ascii")
        self._exec_unchecked(f"echo '{b64}' | base64 -d > {remote_path}")
        logger.debug("Uploaded %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree into the Modal sandbox.

        Creates a tar archive locally, base64-encodes it, and extracts
        it inside the sandbox.
        """
        remote_parent = os.path.dirname(remote_path.rstrip("/"))
        remote_name = os.path.basename(remote_path.rstrip("/"))

        if remote_parent:
            self._exec_unchecked(f"mkdir -p {remote_parent}")

        # Create tar in memory
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(local_path, arcname=remote_name)
        tar_buf.seek(0)

        import base64

        b64 = base64.b64encode(tar_buf.read()).decode("ascii")

        # Write tar to sandbox and extract
        self._exec_unchecked(f"echo '{b64}' | base64 -d | tar xzf - -C {remote_parent}")
        logger.debug("Uploaded dir %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def start_agent_process(self, command: str, port: int) -> None:
        """Start a background process in the Modal sandbox.

        Uses ``nohup`` to background the process and polls a health
        endpoint until it's ready.
        """
        bg_command = f"nohup {command} > /tmp/worker.log 2>&1 &"
        self._exec_unchecked(bg_command)

        self._wait_for_ready(port, timeout=60.0)
        logger.info("Agent process started in sandbox %s on port %d", self.name, port)

    def _wait_for_ready(self, port: int, timeout: float = 60.0) -> None:
        """Poll the health endpoint inside the sandbox."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                process = self._sandbox.exec(
                    "bash",
                    "-c",
                    f"curl -sf http://127.0.0.1:{port}/health",
                )
                process.wait()
                if process.returncode == 0:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"Worker server did not start within {timeout}s in sandbox {self.name}")

    def get_endpoint(self, port: int) -> tuple[str, dict[str, str]]:
        """Return the URL to reach the given port.

        Modal sandboxes use tunnel URLs for external access. For
        internal exec-based access, the loopback address works.
        """
        # For exec-based interaction (our primary use case), loopback works
        return f"http://127.0.0.1:{port}", {}

    def close(self) -> None:
        """Terminate and detach from the Modal sandbox."""
        try:
            self._sandbox.terminate()
        except Exception:
            logger.debug("Sandbox %s terminate error (may already be stopped)", self.name)
        try:
            self._sandbox.detach()
        except Exception:
            pass
        logger.info("ModalSandbox %s closed", self.name)

    def _exec_unchecked(self, command: str) -> str:
        """Execute a command without raising on non-zero exit."""
        try:
            return self.exec(command)
        except RuntimeError:
            return ""


def create_modal_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> ModalSandbox:
    """Factory function for creating a ModalSandbox."""
    return ModalSandbox(name=name, image=image, **kwargs)
