"""Local subprocess sandbox backend for development and testing."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time

logger = logging.getLogger(__name__)


class LocalSandbox:
    """Sandbox implementation using local temp directories and subprocesses.

    Useful for development and testing without Docker or cloud infrastructure.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._workdir = tempfile.mkdtemp(prefix=f"rllm-sandbox-{name}-")
        self._agent_process: subprocess.Popen | None = None
        self._env = os.environ.copy()
        logger.info("LocalSandbox %s created at %s", name, self._workdir)

    def exec(self, command: str, timeout: float | None = None) -> str:
        """Execute a command in the sandbox working directory."""
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

    def start_agent_process(self, command: str, port: int) -> None:
        """Start a background process (e.g. worker_server.py)."""
        translated_cmd = command.replace("/app/", f"{self._workdir}/app/")

        # Set PYTHONPATH so the worker can import from uploaded code
        env = self._env.copy()
        python_paths = [
            os.path.join(self._workdir, "app", "agent"),
            os.path.join(self._workdir, "app", "runner"),
        ]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join(python_paths + ([existing] if existing else []))

        # Redirect stdout/stderr to files to avoid PIPE buffer deadlocks.
        # The subprocess runs an aiohttp server which can produce output at
        # any time; PIPE buffers fill up and block the process if not drained.
        log_dir = os.path.join(self._workdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        stdout_log = open(os.path.join(log_dir, "worker_stdout.log"), "w")
        stderr_log = open(os.path.join(log_dir, "worker_stderr.log"), "w")
        self._log_files = (stdout_log, stderr_log)

        self._agent_process = subprocess.Popen(
            translated_cmd,
            shell=True,
            cwd=self._workdir,
            env=env,
            stdout=stdout_log,
            stderr=stderr_log,
        )

        # Wait for the server to become ready
        self._wait_for_ready(port, timeout=30.0)
        logger.info("Agent process started in sandbox %s (PID: %s, port: %d)", self.name, self._agent_process.pid, port)

    def _wait_for_ready(self, port: int, timeout: float = 30.0) -> None:
        """Poll the /health endpoint until the worker is ready."""
        import urllib.request

        start = time.time()
        url = f"http://127.0.0.1:{port}/health"
        while time.time() - start < timeout:
            try:
                resp = urllib.request.urlopen(url, timeout=1)
                if resp.status == 200:
                    return
            except Exception:
                pass
            time.sleep(0.3)
        raise TimeoutError(f"Worker server did not start within {timeout}s on port {port}")

    def get_endpoint(self, port: int) -> tuple[str, dict[str, str]]:
        """Return the base URL and headers to reach the worker server."""
        return f"http://127.0.0.1:{port}", {}

    def close(self) -> None:
        """Terminate the agent process and clean up the temp directory."""
        if self._agent_process is not None:
            self._agent_process.terminate()
            try:
                self._agent_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._agent_process.kill()
                self._agent_process.wait()
            self._agent_process = None

        # Close log file handles
        for fh in getattr(self, "_log_files", ()):
            try:
                fh.close()
            except Exception:
                pass

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
