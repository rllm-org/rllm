"""Protocol definitions for sandboxed agent execution."""

from __future__ import annotations

import subprocess
from typing import Protocol, runtime_checkable


class SnapshotNotFound(Exception):
    """Raised by ``create_sandbox(image=ref)`` when a snapshot ref no longer
    resolves on its backend, signalling :func:`rllm.sandbox.snapshot.get_sandbox`
    to fall back to the cold path. Transient/auth errors propagate instead.
    """


@runtime_checkable
class Sandbox(Protocol):
    """Protocol for sandbox backends (Docker, Local, Modal, etc.)."""

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        """Execute a command inside the sandbox and return stdout.

        Args:
            command: Shell command to run.
            timeout: Optional per-call timeout (seconds).
            user: Optional UID or username to run the command as. Backends
                that support user isolation (e.g., Docker) should honor this;
                others may ignore it.
        """
        ...

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file into the sandbox."""
        ...

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree into the sandbox."""
        ...

    def close(self) -> None:
        """Destroy the sandbox and release resources."""
        ...


def _safe_exec(sandbox: Sandbox, command: str, timeout: float | None = None) -> str:
    """Execute command, returning stderr on non-zero exit instead of raising."""
    try:
        return sandbox.exec(command, timeout=timeout)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        return str(e)
