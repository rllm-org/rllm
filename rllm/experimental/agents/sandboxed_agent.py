"""SandboxedAgentFlow: base class for agents that need sandboxed execution environments.

Lifecycle managed by EvalRunner:
1. EvalRunner calls setup_sandbox(task, config) — creates Sandbox
2. EvalRunner calls run(task, config) — agent uses self.sandbox
3. EvalRunner calls evaluator.evaluate() — evaluator accesses sandbox via episode artifacts
4. EvalRunner calls teardown_sandbox() — guaranteed cleanup
"""

from __future__ import annotations

import copy
import logging
import subprocess
import uuid
from abc import ABC, abstractmethod

from rllm.experimental.eval.types import AgentConfig, Task
from rllm.sdk.sandbox.protocol import Sandbox
from rllm.types import Episode

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
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def sandbox(self) -> Sandbox | None:
        return self._sandbox

    def create_instance(self) -> SandboxedAgentFlow:
        """Create a per-task copy with fresh sandbox state.

        Called by EvalRunner so each parallel task gets its own sandbox.
        """
        instance = copy.copy(self)
        instance._sandbox = None
        return instance

    def setup_sandbox(self, task: dict, config: AgentConfig) -> None:
        """Create and configure sandbox. Called by EvalRunner before run()."""
        image = self.get_image(task)
        task_id = task.get("instance_id", task.get("task_id", "unknown"))
        name = f"rllm-{task_id}-{uuid.uuid4().hex[:6]}"
        self._sandbox = create_sandbox(self.sandbox_backend, name=name, image=image)

        # Run global setup commands
        for cmd in self.setup_commands:
            _safe_exec(self._sandbox, cmd, timeout=300)

        # Run task-specific setup commands with template interpolation
        for cmd_template in self.task_setup_commands:
            try:
                cmd = cmd_template.format(**task)
            except KeyError:
                cmd = cmd_template
            _safe_exec(self._sandbox, cmd, timeout=300)

        # Subclass hook for additional setup
        self.on_sandbox_ready(task, config)

    def on_sandbox_ready(self, task: dict, config: AgentConfig) -> None:  # noqa: B027
        """Hook for subclasses to run additional setup after sandbox creation."""

    def teardown_sandbox(self) -> None:
        """Destroy sandbox. Called by EvalRunner after evaluate(), even on failure."""
        if self._sandbox is not None:
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
        from rllm.sdk.sandbox.backends.docker import DockerSandbox

        return DockerSandbox(name=name, image=image, **kwargs)
    elif backend == "local":
        from rllm.sdk.sandbox.backends.local import LocalSandbox

        return LocalSandbox(name=name, **kwargs)
    elif backend == "modal":
        from rllm.sdk.sandbox.backends.modal_backend import ModalSandbox

        return ModalSandbox(name=name, **kwargs)
    else:
        raise ValueError(f"Unknown sandbox backend: {backend}. Available: docker, local, modal")


def _safe_exec(sandbox: Sandbox, command: str, timeout: float | None = None) -> str:
    """Execute command, returning stderr on non-zero exit instead of raising."""
    try:
        return sandbox.exec(command, timeout=timeout)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        return str(e)
