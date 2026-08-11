"""Protocol definitions for sandboxed agent execution."""

from __future__ import annotations

import subprocess
from typing import Protocol, runtime_checkable


class SnapshotNotFound(Exception):
    """Raised by ``create_sandbox(image=ref)`` when a snapshot ref no longer
    resolves on its backend, signalling :func:`rllm.sandbox.snapshot.get_sandbox`
    to fall back to the cold path. Transient/auth errors propagate instead.
    """


class SandboxCommandTimeout(RuntimeError):
    """Raised by a backend's ``exec`` when a command is killed for exceeding its
    own ``timeout``. Distinct from a genuine non-zero exit so callers can treat
    "the agent spent its whole time budget" as expected, not a failure.
    Subclasses ``RuntimeError`` so existing handlers keep catching it.
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

    def download_file(self, remote_path: str) -> bytes:
        """Read a file out of the sandbox.

        The counterpart to :meth:`upload_file`, needed whenever something has to
        leave a sandbox — collecting a separate-mode verifier's artifacts, for
        one. Backends use their native transfer (Modal's filesystem API, Daytona's
        ``fs``, ``docker cp``) rather than shelling out, so binary payloads survive
        intact. Raises ``FileNotFoundError`` when the path isn't there.
        """
        ...

    def close(self) -> None:
        """Destroy the sandbox and release resources."""
        ...

    def is_alive(self) -> bool:
        """Return whether the sandbox is still usable, via a cheap backend API query.

        Remote sandboxes can die out-of-band (provider idle auto-stop,
        lifetime timeout, external deletion); callers that hold a sandbox
        without using it (e.g. the warm queue) check this before handing
        it to a consumer. Implementations must not raise: any error in
        the check means ``False``.
        """
        ...


@runtime_checkable
class ComposeSandbox(Sandbox, Protocol):
    """A :class:`Sandbox` backed by a Docker Compose project.

    The ``Sandbox`` methods address the project's ``main`` service; these
    extensions reach the task's sidecar services (databases, APIs, event
    feeds) by Compose service name. Like ``Sandbox`` itself this is a
    structural contract — implementations (``DockerComposeSandbox``,
    ``ModalComposeSandbox``) satisfy it without inheriting it.

    The contract describes *methods*, not interchangeability: implementations
    may differ in capability (e.g. Modal's DinD VM runtime cannot pass a GPU
    through, so ``ModalComposeSandbox`` rejects ``gpu`` at construction while
    ``DockerComposeSandbox`` accepts it).
    """

    def set_env(self, env: dict[str, str]) -> None:
        """Persist environment variables for subsequent ``exec`` calls on ``main``."""
        ...

    def service_exec(self, service: str, command: str, timeout: float | None = None, user: str | None = None) -> str:
        """Run *command* inside a named Compose service and return stdout."""
        ...

    def service_download_file(self, service: str, remote_path: str) -> bytes:
        """Read a file out of a named Compose service."""
        ...

    def stop_service(self, service: str) -> None:
        """Stop one Compose service (e.g. freeze ``main`` before sidecar collection)."""
        ...


def _safe_exec(sandbox: Sandbox, command: str, timeout: float | None = None) -> str:
    """Execute command, returning stderr on non-zero exit instead of raising."""
    try:
        return sandbox.exec(command, timeout=timeout)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        return str(e)
