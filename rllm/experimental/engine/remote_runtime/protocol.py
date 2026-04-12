"""Remote agent runtime abstractions for use cases where agent and environment colocate in the container"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from rllm.workflows.workflow import TerminationReason

# ---------------------------------------------------------------------------
# Common protocol (backend-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class RemoteRuntimeConfig:
    """Common config for all remote runtimes."""

    enabled: bool = False
    backend: str = "agentcore"
    agentcore: dict[str, Any] = field(default_factory=dict)
    harbor: dict[str, Any] = field(default_factory=dict)
    session_timeout: float = 900.0


@dataclass
class TaskSubmission:
    """A single task to submit to a remote runtime."""

    task: dict
    session_id: str
    task_id: str  # GRPO grouping key (maps to ART input_id)
    inference_url: str  # Per-session gateway URL


@dataclass
class RemoteTaskResult:
    """Result returned from a remote runtime."""

    finished: bool  # True if agent loop completed; False on transport/application error
    session_id: str
    task_id: str = ""  # GRPO grouping key (from TaskSubmission)
    reward: float | None = None
    error: str | None = None
    termination_reason: TerminationReason | None = None
    elapsed: float = 0.0
    raw_result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RemoteAgentRuntime(Protocol):
    """Protocol for Pattern 1 remote agent runtimes."""

    def initialize(self) -> None:
        """Client setup from config."""
        ...

    async def execute_tasks(self, submissions: list[TaskSubmission], timeout: float | None = None) -> list[RemoteTaskResult]:
        """Submit tasks concurrently and gather results. Returns one result per submission."""
        ...

    def shutdown(self) -> None:
        """Cleanup resources."""
        ...


# ---------------------------------------------------------------------------
# AgentCore-specific config
# ---------------------------------------------------------------------------


@dataclass
class AgentCoreRuntimeConfig:
    """Config specific to AWS Bedrock AgentCore Runtime."""

    agent_runtime_arn: str = ""
    s3_bucket: str = ""
    tps_limit: int = 25


# ---------------------------------------------------------------------------
# Harbor-specific config
# ---------------------------------------------------------------------------


@dataclass
class HarborRuntimeConfig:
    """Config specific to harbor-backed trials."""

    agent: str = "mini-swe-agent"
    # Harbor environment backend: "docker" | "daytona" | "modal" | "e2b" | "runloop" | "gke" | "apple-container".
    # None → use the task's declared default.
    environment_type: str | None = None
    # Pass-through to harbor AgentConfig.kwargs (scaffold-specific flags).
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    # Per-stage timeout multipliers — None means use the task's default (multiplier 1.0).
    agent_timeout_multiplier: float | None = None
    verifier_timeout_multiplier: float | None = None
    agent_setup_timeout_multiplier: float | None = None
    environment_build_timeout_multiplier: float | None = None
