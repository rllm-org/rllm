"""Daytona sandbox backend.

Uses Daytona Cloud Sandboxes (https://www.daytona.io/docs) to run agent
code in remote cloud containers.

Requires the ``daytona`` package::

    pip install daytona

Authentication is handled via the ``DAYTONA_API_KEY``, ``DAYTONA_API_URL``,
and ``DAYTONA_TARGET`` environment variables, or by passing an explicit
``DaytonaConfig`` via the ``daytona_config`` kwarg.
"""

from __future__ import annotations

import atexit
import io
import logging
import os
import tarfile
import threading
import time
import weakref
from pathlib import Path

logger = logging.getLogger(__name__)

# Default auto-stop interval (minutes). 0 disables auto-stop entirely;
# we rely on explicit ``close()`` plus the atexit hook below. Set to a
# finite value to belt-and-suspenders against leaked sandboxes if the
# process is SIGKILL'd (atexit won't fire then).
_DEFAULT_AUTO_STOP_INTERVAL = 30  # minutes; mirrors ModalSandbox's 30-min timeout

# atexit-tracked sandboxes; terminated on process exit to avoid leaks.
_LIVE_SANDBOXES: weakref.WeakSet = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


def _terminate_all_live() -> None:
    """atexit hook: delete every still-alive DaytonaSandbox."""
    with _LIVE_LOCK:
        survivors = list(_LIVE_SANDBOXES)
    if not survivors:
        return
    logger.warning("atexit: deleting %d unreleased Daytona sandbox(es)", len(survivors))
    for sb in survivors:
        try:
            sb.close()
        except Exception:
            logger.debug("atexit: error closing %s", getattr(sb, "name", "<unknown>"), exc_info=True)


atexit.register(_terminate_all_live)


def _looks_like_docker_image(image: str) -> bool:
    """Heuristic: treat strings containing ``:`` or ``/`` as Docker image refs.

    ``python:3.11-slim`` / ``ghcr.io/foo/bar:tag`` → Docker image (Daytona
    will pull from a registry). ``rllm-worker-v0`` → Daytona snapshot name.
    """
    return ":" in image or "/" in image


class DaytonaSandbox:
    """Sandbox implementation using Daytona Cloud Sandboxes.

    Creates a Daytona Sandbox via ``daytona.create()``, executes commands
    via ``sandbox.process.exec()``, uploads files via the native
    ``sandbox.fs`` API, and exposes ports via ``sandbox.get_preview_link()``.

    The ``image`` parameter accepts either:
    - A Daytona snapshot name (e.g. ``"rllm-worker-v0"``) — passed via
      ``CreateSandboxFromSnapshotParams``.
    - A Docker image reference (e.g. ``"python:3.11-slim"``) — passed via
      ``CreateSandboxFromImageParams`` so Daytona pulls it from the
      registry.
    - A ``daytona.Image`` object — used directly as a custom build spec.

    Optional kwargs:
    - ``daytona_config``: explicit ``DaytonaConfig`` (overrides env vars).
    - ``auto_stop_interval``: minutes; default ``30``. Pass ``0`` to disable.
    - ``auto_archive_interval``: minutes; default Daytona's own default.
    - ``cpu`` / ``memory`` / ``disk`` / ``gpu``: resource overrides (only
      effective with the from-image path).
    - ``env_vars``: dict of env vars baked into the sandbox.
    - ``labels``: dict of labels for filtering / billing.
    - ``os_user``: default user inside the sandbox.
    - ``create_timeout``: seconds to wait for ``create()``; default ``120``.
    """

    def __init__(self, name: str, image: str = "python:3.11-slim", **kwargs):
        # Lazy import so users without the SDK can still import this module.
        from daytona import (
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            Daytona,
            Image,
            Resources,
        )

        self.name = name
        self._image_spec = image
        self._closed = False
        self._worker_session_id: str | None = None
        self._auto_stop_interval = int(kwargs.pop("auto_stop_interval", _DEFAULT_AUTO_STOP_INTERVAL))
        self._auto_archive_interval = kwargs.pop("auto_archive_interval", None)
        self._create_timeout = float(kwargs.pop("create_timeout", 120.0))

        # Client init
        daytona_config = kwargs.pop("daytona_config", None)
        self._client = Daytona(daytona_config) if daytona_config is not None else Daytona()

        # Build common parameters
        base_kwargs: dict = {
            "name": name,
            "auto_stop_interval": self._auto_stop_interval,
        }
        if self._auto_archive_interval is not None:
            base_kwargs["auto_archive_interval"] = int(self._auto_archive_interval)
        for key in ("env_vars", "labels", "os_user", "public", "volumes", "auto_delete_interval"):
            if key in kwargs:
                base_kwargs[key] = kwargs.pop(key)

        # Resources only apply to the from-image path
        resources = None
        resource_keys = {"cpu", "memory", "disk", "gpu"}
        if resource_keys & kwargs.keys():
            resources = Resources(**{k: kwargs.pop(k) for k in list(kwargs.keys()) if k in resource_keys})

        # Decide create path: snapshot name vs Docker image vs Image object
        if isinstance(image, Image):
            params = CreateSandboxFromImageParams(image=image, resources=resources, **base_kwargs)
            self._sandbox = self._client.create(params, timeout=self._create_timeout)
            image_label = "<daytona.Image>"
        elif isinstance(image, str) and _looks_like_docker_image(image):
            params = CreateSandboxFromImageParams(image=image, resources=resources, **base_kwargs)
            self._sandbox = self._client.create(params, timeout=self._create_timeout)
            image_label = image
        else:
            # Treat as Daytona snapshot name
            if resources is not None:
                logger.warning(
                    "Resources are ignored when creating from a snapshot (%s); snapshot resources win.",
                    image,
                )
            params = CreateSandboxFromSnapshotParams(snapshot=image, **base_kwargs)
            self._sandbox = self._client.create(params, timeout=self._create_timeout)
            image_label = f"snapshot:{image}"

        self._sandbox_id = getattr(self._sandbox, "id", None) or getattr(self._sandbox, "sandbox_id", None)

        with _LIVE_LOCK:
            _LIVE_SANDBOXES.add(self)

        logger.info(
            "DaytonaSandbox %s created (id: %s, image: %s)",
            name,
            self._sandbox_id,
            image_label,
        )

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:  # noqa: ARG002
        """Execute a command inside the Daytona sandbox.

        ``user`` is accepted for protocol compatibility but ignored —
        Daytona sets the default user at sandbox creation time
        (``os_user`` kwarg), not per exec.

        Wraps the command in ``bash -c`` for shell-feature parity with
        DockerSandbox / ModalSandbox. Returns stdout. Raises
        ``RuntimeError`` on non-zero exit.
        """
        exec_kwargs: dict = {}
        if timeout is not None:
            exec_kwargs["timeout"] = int(timeout)

        response = self._sandbox.process.exec(f"bash -c {_shell_quote(command)}", **exec_kwargs)

        stdout = response.result or ""
        exit_code = response.exit_code

        if exit_code != 0:
            tail = stdout[-500:]
            logger.warning(
                "Command failed in sandbox %s: %s\noutput tail: %s",
                self.name,
                command,
                tail,
            )
            raise RuntimeError(f"Command failed (exit {exit_code}) in sandbox {self.name}: {command}\n{tail}")
        return stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file via Daytona's native ``fs.upload_file``."""
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            self._exec_unchecked(f"mkdir -p {remote_dir}")
        self._sandbox.fs.upload_file(local_path, remote_path)
        logger.debug("Uploaded %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree.

        Daytona's ``fs`` API has no native recursive upload, so we package
        the tree into a single ``.tar.gz`` locally, upload that one file
        with the native API, then extract it inside the sandbox. One HTTP
        upload + one exec, regardless of tree size.
        """
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"upload_dir: local path {local_path} does not exist")

        remote_parent = os.path.dirname(remote_path.rstrip("/")) or "/"
        remote_name = os.path.basename(remote_path.rstrip("/")) or local.name

        self._exec_unchecked(f"mkdir -p {remote_parent}")

        # Build tar.gz in memory, write to a temp file, upload, extract.
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(local_path, arcname=remote_name)
        tar_buf.seek(0)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(tar_buf.read())
            tmp_path = tmp.name

        try:
            remote_tar = f"/tmp/_upload_{self.name}_{int(time.time() * 1000)}.tar.gz"
            self._sandbox.fs.upload_file(tmp_path, remote_tar)
            self.exec(f"tar xzf {remote_tar} -C {remote_parent} && rm -f {remote_tar}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        logger.debug("Uploaded dir %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def start_agent_process(self, command: str, port: int) -> None:
        """Start a long-running process (worker_server.py) and poll until ready.

        Daytona's ephemeral ``process.exec`` reaps backgrounded children
        when the call returns, so ``nohup ... &`` (the Modal/Docker
        pattern) doesn't survive here. We launch inside a persistent
        ``process.create_session`` instead — the subprocess lives until
        the session is deleted, which ``close()`` handles.
        """
        from daytona import SessionExecuteRequest

        session_id = f"rllm-worker-{self.name}"
        self._sandbox.process.create_session(session_id)
        self._worker_session_id = session_id
        self._sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(command=command, run_async=True),
        )
        self._wait_for_ready(port, timeout=60.0)
        logger.info("Agent process started in sandbox %s on port %d (session: %s)", self.name, port, session_id)

    def _wait_for_ready(self, port: int, timeout: float = 60.0) -> None:
        """Poll the in-sandbox health endpoint until 200 or timeout.

        Uses ``python3 + urllib`` because slim Python images don't ship
        ``curl`` or ``wget``.
        """
        health_cmd = f"python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=2)\""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = self._sandbox.process.exec(health_cmd)
                if response.exit_code == 0:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"Worker server did not start within {timeout}s in sandbox {self.name}")

    def get_endpoint(self, port: int) -> tuple[str, dict[str, str]]:
        """Return a public URL + headers to reach the given in-sandbox port.

        Uses Daytona's native preview-link API, so the caller can reach
        the worker_server from anywhere on the network (no cloudflared
        tunnel needed for this direction).
        """
        preview = self._sandbox.get_preview_link(port=port)
        return preview.url, {"x-daytona-preview-token": preview.token}

    def close(self) -> None:
        """Delete the Daytona sandbox and release resources.

        Deletes the worker session first (which kills the long-running
        worker_server process), then deletes the sandbox itself.
        """
        if self._closed:
            return
        if self._worker_session_id is not None:
            try:
                self._sandbox.process.delete_session(self._worker_session_id)
            except Exception:
                logger.debug("Sandbox %s: worker session delete error", self.name, exc_info=True)
            self._worker_session_id = None
        try:
            self._sandbox.delete()
        except Exception:
            logger.debug("Sandbox %s delete error (may already be gone)", self.name, exc_info=True)
        with _LIVE_LOCK:
            _LIVE_SANDBOXES.discard(self)
        self._closed = True
        logger.info("DaytonaSandbox %s closed", self.name)

    def _exec_unchecked(self, command: str) -> str:
        """Execute a command without raising on non-zero exit."""
        try:
            return self.exec(command)
        except RuntimeError:
            return ""


def _shell_quote(s: str) -> str:
    """Single-quote a string for safe interpolation into a shell command."""
    return "'" + s.replace("'", "'\\''") + "'"


def create_daytona_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> DaytonaSandbox:
    """Factory function for creating a DaytonaSandbox."""
    return DaytonaSandbox(name=name, image=image, **kwargs)
