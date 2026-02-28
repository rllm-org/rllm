"""Protocol definitions for sandboxed agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SandboxConfig:
    """Configuration for sandboxed agent execution."""

    enabled: bool = False
    backend: str = "local"  # local | docker | modal | agentcore
    agent_dir: str = ""
    agent_module: str = "agent"
    agent_func: str = "rollout"
    image: str = "python:3.11-slim"
    dockerfile: str = ""
    requirements_file: str = ""
    install_rllm_sdk: bool = True

    # Orchestration mode
    pool_mode: str = "persistent"  # persistent | per_task

    # Persistent mode settings
    num_workers: int = 8
    worker_port: int = 8100

    # Per-task mode settings
    max_concurrent: int = 32
    task_setup_commands: list[str] = field(default_factory=list)
    task_setup_timeout: int = 300
    teardown_on_complete: bool = True

    # Execution settings
    execution_timeout: float = 600.0
    setup_commands: list[str] = field(default_factory=list)

    # Extra backend-specific config
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of a sandboxed agent execution."""

    success: bool
    trajectories: list[dict] | None = None
    session_uid: str = ""
    reward: float | None = None
    error: str | None = None
    elapsed: float = 0.0


@runtime_checkable
class Sandbox(Protocol):
    """Protocol for sandbox backends (Docker, Local, Modal, etc.)."""

    def exec(self, command: str, timeout: float | None = None) -> str:
        """Execute a command inside the sandbox and return stdout."""
        ...

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file into the sandbox."""
        ...

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree into the sandbox."""
        ...

    def start_agent_process(self, command: str, port: int) -> None:
        """Start a long-running process (e.g. worker_server.py) in the background."""
        ...

    def get_endpoint(self, port: int) -> tuple[str, dict[str, str]]:
        """Return (base_url, headers) to reach the given port inside the sandbox."""
        ...

    def close(self) -> None:
        """Destroy the sandbox and release resources."""
        ...


# Type alias for sandbox factory callables
SandboxFactory = type[Sandbox] | Any  # Callable that returns a Sandbox
