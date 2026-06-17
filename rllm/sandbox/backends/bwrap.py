"""Bubblewrap sandbox backend for Koala GPU cluster.

Uses user namespaces (--unshare-user) which work in Kubernetes pods
without CAP_SYS_ADMIN. Provides filesystem and network isolation.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)

_BWRAP = "bwrap"

# Paths to mount read-only. On merged-usr systems (modern distros, Koala pods)
# /bin, /lib, /lib64, /sbin are symlinks to /usr/...; we detect and use --symlink.
_RO_BIND_PATHS = ["/usr", "/etc/alternatives", "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d"]
_SYMLINK_CANDIDATES = [("/bin", "/usr/bin"), ("/lib", "/usr/lib"), ("/lib64", "/usr/lib64"), ("/sbin", "/usr/sbin")]


class BwrapSandbox:
    """Sandbox using bubblewrap for process isolation.

    Only whitelisted host paths are mounted read-only; a private tmpfs
    provides /tmp. Network is blocked via --unshare-net.
    """

    def __init__(self, name: str, network: bool = False, **kwargs):
        self.name = name
        self._network = network
        self._workdir = tempfile.mkdtemp(prefix=f"rllm-bwrap-{name}-")
        self._app_dir = os.path.join(self._workdir, "app")
        self._tmp_dir = os.path.join(self._workdir, "tmp")
        os.makedirs(self._app_dir, exist_ok=True)
        os.makedirs(self._tmp_dir, exist_ok=True)
        self._cmd_prefix = self._build_static_prefix()
        logger.info("BwrapSandbox %s created at %s", name, self._workdir)

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:  # noqa: ARG002
        timeout = timeout if timeout is not None else 30.0
        cmd = self._build_bwrap_cmd(command)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("bwrap exec failed (exit %d): %s", result.returncode, command[:200])
            raise subprocess.CalledProcessError(
                result.returncode, command, result.stdout, result.stderr
            )
        return result.stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        dest = self._translate_path(remote_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(local_path, dest)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        dest = self._translate_path(remote_path)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(local_path, dest)

    def close(self) -> None:
        if os.path.exists(self._workdir):
            shutil.rmtree(self._workdir, ignore_errors=True)
        logger.info("BwrapSandbox %s closed", self.name)

    def is_alive(self) -> bool:
        try:
            return os.path.exists(self._workdir)
        except OSError:
            return False

    def _build_static_prefix(self) -> list[str]:
        cmd = [
            _BWRAP,
            "--unshare-user",
            "--uid", "1000",
            "--gid", "1000",
        ]
        if not self._network:
            cmd.append("--unshare-net")
        for path in _RO_BIND_PATHS:
            if os.path.exists(path):
                cmd.extend(["--ro-bind", path, path])
        for link_path, target in _SYMLINK_CANDIDATES:
            if os.path.islink(link_path):
                cmd.extend(["--symlink", target, link_path])
            elif os.path.isdir(link_path):
                cmd.extend(["--ro-bind", link_path, link_path])
        cmd.extend([
            "--tmpfs", "/root",
            "--dev", "/dev",
            "--proc", "/proc",
            "--bind", self._tmp_dir, "/tmp",
            "--bind", self._app_dir, "/app",
            "--die-with-parent",
        ])
        return cmd

    def _build_bwrap_cmd(self, command: str) -> list[str]:
        return self._cmd_prefix + ["--", "bash", "-c", command]

    def _translate_path(self, remote_path: str) -> str:
        """Translate sandbox-visible paths to host workdir paths.

        Only /tmp and /app are writable in the sandbox, so we validate
        that uploads target one of those prefixes.
        """
        if remote_path.startswith("/tmp/") or remote_path == "/tmp":
            rel = remote_path[len("/tmp"):].lstrip("/")
            result = os.path.realpath(os.path.join(self._tmp_dir, rel))
            if not result.startswith(os.path.realpath(self._tmp_dir)):
                raise ValueError(f"Path escapes sandbox: {remote_path}")
            return result
        elif remote_path.startswith("/app/") or remote_path == "/app":
            rel = remote_path[len("/app"):].lstrip("/")
            result = os.path.realpath(os.path.join(self._app_dir, rel))
            if not result.startswith(os.path.realpath(self._app_dir)):
                raise ValueError(f"Path escapes sandbox: {remote_path}")
            return result
        elif not remote_path.startswith("/"):
            result = os.path.realpath(os.path.join(self._app_dir, remote_path))
            if not result.startswith(os.path.realpath(self._app_dir)):
                raise ValueError(f"Path escapes sandbox: {remote_path}")
            return result
        else:
            raise ValueError(
                f"Upload path must be under /tmp or /app (got: {remote_path}). "
                "Only these directories are writable inside the bwrap sandbox."
            )


def is_available() -> bool:
    """Check if bwrap is installed on the system."""
    return shutil.which(_BWRAP) is not None
