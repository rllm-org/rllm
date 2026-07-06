"""Local subprocess sandbox backend for development and testing."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class LocalSandbox:
    """Sandbox implementation using local temp directories and subprocesses.

    Useful for development and testing without Docker or cloud infrastructure.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._workdir = tempfile.mkdtemp(prefix=f"rllm-sandbox-{name}-")
        self._env = os.environ.copy()
        logger.info("LocalSandbox %s created at %s", name, self._workdir)

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:  # noqa: ARG002
        """Execute a command in the sandbox working directory.

        ``user`` is accepted for protocol compatibility but ignored — the
        local backend always runs as the host user.
        """
        timeout = timeout or 300.0
        # Translate absolute /app/ paths to the workdir
        translated_cmd = command.replace("/app/", f"{self._workdir}/app/")
        result = subprocess.run(
            translated_cmd,
            shell=True,
            cwd=self._workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._env,
        )
        if result.returncode != 0:
            logger.warning("Command failed in sandbox %s: %s\nstderr: %s", self.name, translated_cmd, result.stderr[:500])
            raise subprocess.CalledProcessError(result.returncode, translated_cmd, result.stdout, result.stderr)
        return result.stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Copy a file into the sandbox working directory."""
        dest = self._translate_path(remote_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(local_path, dest)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Copy a directory tree into the sandbox working directory."""
        dest = self._translate_path(remote_path)
        if os.path.exists(dest):
            shutil.rmtree(dest)

        # Skip heavy directories that are never needed inside a sandbox
        _IGNORE_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", ".tox", ".mypy_cache", ".pytest_cache", "*.egg-info"}

        def _ignore(directory: str, contents: list[str]) -> set[str]:
            return {c for c in contents if c in _IGNORE_DIRS or c.endswith(".egg-info")}

        shutil.copytree(local_path, dest, ignore=_ignore)

    def is_alive(self) -> bool:
        """The sandbox is just a temp working directory; alive while it exists."""
        return os.path.isdir(self._workdir)

    def close(self) -> None:
        """Clean up the temp directory."""
        if os.path.exists(self._workdir):
            shutil.rmtree(self._workdir, ignore_errors=True)
        logger.info("LocalSandbox %s closed", self.name)

    def _translate_path(self, remote_path: str) -> str:
        """Translate an absolute /app/... path to the local workdir."""
        if remote_path.startswith("/"):
            # Strip leading / and join with workdir
            return os.path.join(self._workdir, remote_path.lstrip("/"))
        return os.path.join(self._workdir, remote_path)


def create_local_sandbox(name: str, **kwargs) -> LocalSandbox:
    """Factory function for creating a LocalSandbox."""
    return LocalSandbox(name=name, **kwargs)
