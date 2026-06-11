"""SandboxedAgentFlow: base class for agents that need sandboxed execution environments.

Lifecycle managed by :class:`rllm.hooks.SandboxTaskHooks` (which sits in
front of :class:`rllm.engine.agentflow_engine.AgentFlowEngine`):

1. The hook creates a Sandbox via ``_create_sandbox_for_task`` and injects
   it with ``set_sandbox()``, then calls ``on_sandbox_ready(task, config)``.
2. The engine calls ``run(task, config)`` — the agent uses ``self.sandbox``.
3. The hook resolves an Evaluator from the Task's verifier config (or uses
   ``evaluator_override``) and runs ``evaluator.evaluate(task, episode)``.
4. The hook's teardown closure calls ``teardown_sandbox()`` — guaranteed
   cleanup, no-op when the sandbox was injected externally.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod

from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Episode, Task

logger = logging.getLogger(__name__)


class SandboxedAgentFlow(ABC):
    """Base class for agents that need sandboxed execution environments.

    The sandbox backend is pluggable via ``sandbox_backend``:
    ``"docker"`` | ``"local"`` | ``"modal"`` | ``"daytona"``.

    Subclasses must implement :meth:`run` and may override
    :meth:`get_image` or :meth:`on_sandbox_ready` for task-specific setup.
    """

    sandbox_backend: str = "docker"
    image: str = "python:3.11-slim"
    max_concurrent: int = 4

    def __init__(self, **kwargs):
        self._sandbox: Sandbox | None = None
        self._sandbox_externally_managed: bool = False
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def sandbox(self) -> Sandbox | None:
        return self._sandbox

    def set_sandbox(self, sandbox: Sandbox) -> None:
        """Inject a sandbox managed by an external orchestrator (e.g. ``Runner``).

        The agent flow will use this sandbox but will not close it; the
        orchestrator owns the lifecycle. ``teardown_sandbox`` becomes a no-op.
        """
        self._sandbox = sandbox
        self._sandbox_externally_managed = True

    def create_instance(self) -> SandboxedAgentFlow:
        """Create a per-task copy with fresh sandbox state.

        Called by :func:`rllm.eval.runner.run_dataset` so each parallel
        task gets its own sandbox.
        """
        instance = copy.copy(self)
        instance._sandbox = None
        instance._sandbox_externally_managed = False
        return instance

    def on_sandbox_ready(self, task: dict, config: AgentConfig) -> None:  # noqa: B027
        """Hook for subclasses to run additional setup after sandbox creation."""

    def teardown_sandbox(self) -> None:
        """Destroy sandbox. Called by the Runner after evaluate(), even on failure.

        No-op when the sandbox was injected externally (the orchestrator owns it).
        """
        if self._sandbox is None:
            return
        if self._sandbox_externally_managed:
            self._sandbox = None
            return
        try:
            self._sandbox.close()
        except Exception:
            logger.exception("Sandbox teardown error")
        self._sandbox = None

    def get_image(self, task: dict) -> str:
        """Return container image for this task. Override for per-task images."""
        return self.image

    @abstractmethod
    def run(self, task: Task, config: AgentConfig) -> Episode: ...


def create_sandbox(backend: str, name: str, image: str, **kwargs) -> Sandbox:
    """Factory: create a Sandbox from a backend name. Lazy imports."""
    if backend == "docker":
        from rllm.sandbox.backends.docker import DockerSandbox

        return DockerSandbox(name=name, image=image, **kwargs)
    elif backend == "local":
        from rllm.sandbox.backends.local import LocalSandbox

        return LocalSandbox(name=name, **kwargs)
    elif backend == "modal":
        from rllm.sandbox.backends.modal_backend import ModalSandbox

        return ModalSandbox(name=name, image=image, **kwargs)
    elif backend == "daytona":
        from rllm.sandbox.backends.daytona import DaytonaSandbox

        return DaytonaSandbox(name=name, image=image, **kwargs)
    else:
        raise ValueError(f"Unknown sandbox backend: {backend}. Available: docker, local, modal, daytona")


def build_snapshot(backend: str, task: Task, key: str, prior_ref: str | None = None, *, force: bool = False) -> str | None:
    """Build a snapshot of ``task``'s environment; return a backend ref, or ``None``.

    Each backend owns its mechanism (Modal: live-FS capture; Daytona:
    declarative bake). Backends without snapshots (docker/local) return ``None``.
    A known-live ``prior_ref`` is reused unless ``force``, which always rebuilds.
    """
    if backend == "modal":
        from rllm.sandbox.backends.modal_backend import build_modal_snapshot

        return build_modal_snapshot(task, key, prior_ref, force=force)
    elif backend == "daytona":
        from rllm.sandbox.backends.daytona import build_daytona_snapshot

        return build_daytona_snapshot(task, key, force=force)
    return None


def delete_snapshot(backend: str, ref: str) -> bool:
    """Delete a snapshot from its backend. Returns ``True`` on success."""
    if backend == "modal":
        from rllm.sandbox.backends.modal_backend import delete_modal_snapshot

        return delete_modal_snapshot(ref)
    elif backend == "daytona":
        from rllm.sandbox.backends.daytona import delete_daytona_snapshot

        return delete_daytona_snapshot(ref)
    return False


def snapshot_absent(backend: str, ref: str) -> bool:
    """No-boot probe for ``registry.sync``: ``True`` only when ``ref`` is verifiably gone.

    Conservative by construction — auth/permission/rate-limit/unknown errors return
    ``False`` so sync never prunes a record it cannot confirm is absent.
    """
    if backend == "modal":
        from rllm.sandbox.backends.modal_backend import _modal_ref_absent

        return _modal_ref_absent(ref)
    elif backend == "daytona":
        from rllm.sandbox.backends.daytona import _daytona_ref_absent

        return _daytona_ref_absent(ref)
    return False
