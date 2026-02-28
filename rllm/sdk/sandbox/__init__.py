"""Sandboxed agent execution for rLLM SDK training."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllm.sdk.sandbox.orchestrator import SandboxOrchestrator
    from rllm.sdk.sandbox.protocol import SandboxConfig


def create_sandbox_orchestrator(config: SandboxConfig) -> SandboxOrchestrator:
    """Create a SandboxOrchestrator with the appropriate backend factory.

    Lazily imports backend modules to avoid pulling in optional dependencies
    (e.g. ``docker``, ``modal``) when they aren't needed.
    """
    from rllm.sdk.sandbox.orchestrator import SandboxOrchestrator

    backend = config.backend

    if backend == "local":
        from rllm.sdk.sandbox.backends.local import create_local_sandbox

        factory = create_local_sandbox
    elif backend == "docker":
        from rllm.sdk.sandbox.backends.docker import create_docker_sandbox

        factory = create_docker_sandbox
    elif backend == "modal":
        from rllm.sdk.sandbox.backends.modal_backend import create_modal_sandbox

        factory = create_modal_sandbox
    elif backend == "agentcore":
        from rllm.sdk.sandbox.backends.agentcore import create_agentcore_sandbox

        factory = create_agentcore_sandbox
    else:
        raise ValueError(f"Unknown sandbox backend: {backend!r}. Supported: local, docker, modal, agentcore")

    return SandboxOrchestrator(sandbox_factory=factory, config=config)
