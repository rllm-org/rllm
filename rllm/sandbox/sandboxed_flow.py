"""SandboxedAgentFlow: base class for agents that need sandboxed execution environments.

Lifecycle managed by :class:`rllm.runner.Runner`:

1. Runner creates a Sandbox via ``_create_sandbox_for_task`` and injects it
   with ``set_sandbox()``, then calls ``on_sandbox_ready(task, config)``.
2. Runner calls ``run(task, config)`` — the agent uses ``self.sandbox``.
3. Runner resolves an Evaluator from the Task's verifier config (or uses
   ``evaluator_override``) and runs ``evaluator.evaluate(task, episode)``.
4. Runner calls ``teardown_sandbox()`` — guaranteed cleanup, no-op when the
   sandbox was injected externally.
"""

from __future__ import annotations

import copy
import logging
import uuid
from abc import ABC, abstractmethod

from rllm.sandbox.protocol import Sandbox, _safe_exec
from rllm.types import AgentConfig, Episode, Task

logger = logging.getLogger(__name__)


class SandboxedAgentFlow(ABC):
    """Base class for agents that need sandboxed execution environments.

    The sandbox backend is pluggable via ``sandbox_backend``:
    ``"docker"`` | ``"local"`` | ``"modal"``.

    Subclasses must implement :meth:`run` and may override
    :meth:`get_image` or :meth:`on_sandbox_ready` for task-specific setup.
    """

    sandbox_backend: str = "docker"
    image: str = "python:3.11-slim"
    max_concurrent: int = 4
    setup_commands: list[str] = []
    task_setup_commands: list[str] = []

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

    def setup_sandbox(self, task: dict, config: AgentConfig) -> None:
        """Create and configure sandbox.

        Legacy entry point retained for callers that still build a sandbox
        inside the agent flow. The :class:`rllm.runner.Runner` path uses
        :meth:`set_sandbox` + :meth:`on_sandbox_ready` instead.
        """
        image = self.get_image(task)
        task_id = task.get("instance_id", task.get("task_id", "unknown"))
        name = f"rllm-{task_id}-{uuid.uuid4().hex[:6]}"
        self._sandbox = create_sandbox(self.sandbox_backend, name=name, image=image)

        for cmd in self.setup_commands:
            _safe_exec(self._sandbox, cmd, timeout=300)

        for cmd_template in self.task_setup_commands:
            try:
                cmd = cmd_template.format(**task)
            except KeyError:
                cmd = cmd_template
            _safe_exec(self._sandbox, cmd, timeout=300)

        self.on_sandbox_ready(task, config)

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

        return ModalSandbox(name=name, **kwargs)
    else:
        raise ValueError(f"Unknown sandbox backend: {backend}. Available: docker, local, modal")
