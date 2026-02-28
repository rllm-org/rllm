"""Sandboxed agent execution for rLLM SDK training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rllm.sdk.sandbox.protocol import SandboxConfig


def create_sandbox_orchestrator(config: SandboxConfig) -> Any:
    """Create a sandbox orchestrator with the appropriate backend.

    Lazily imports backend modules to avoid pulling in optional dependencies
    (e.g. ``docker``, ``modal``, ``boto3``) when they aren't needed.

    For **agentcore**, returns an ``AgentCoreOrchestrator`` directly — it
    bypasses the ``SandboxOrchestrator`` / ``Sandbox`` protocol because ACR
    agents are pre-deployed containers invoked via ``boto3`` rather than
    managed sandboxes.

    All other backends return a ``SandboxOrchestrator`` wrapping a backend-
    specific ``Sandbox`` factory.
    """
    backend = config.backend

    # AgentCore has its own orchestrator — no Sandbox protocol needed.
    if backend == "agentcore":
        from rllm.sdk.sandbox.backends.agentcore import AgentCoreOrchestrator

        return AgentCoreOrchestrator(config=config)

    # All other backends use the standard SandboxOrchestrator.
    from rllm.sdk.sandbox.orchestrator import SandboxOrchestrator

    if backend == "local":
        from rllm.sdk.sandbox.backends.local import create_local_sandbox

        factory = create_local_sandbox
    elif backend == "docker":
        from rllm.sdk.sandbox.backends.docker import create_docker_sandbox

        factory = create_docker_sandbox
    elif backend == "modal":
        from rllm.sdk.sandbox.backends.modal_backend import create_modal_sandbox

        factory = create_modal_sandbox
    else:
        raise ValueError(f"Unknown sandbox backend: {backend!r}. Supported: local, docker, modal, agentcore")

    return SandboxOrchestrator(sandbox_factory=factory, config=config)
