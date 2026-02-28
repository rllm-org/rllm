"""AWS Bedrock AgentCore sandbox backend (stub).

Implement this backend to run agent code on AgentCore managed runtimes.
"""

from __future__ import annotations


class AgentCoreSandbox:
    """Stub for AWS Bedrock AgentCore sandbox backend."""

    def __init__(self, name: str, **kwargs):
        raise NotImplementedError(
            "AgentCore sandbox backend is not yet implemented. "
            "To use AgentCore runtimes, install the AWS SDK and "
            "implement AgentCoreSandbox in rllm/sdk/sandbox/backends/agentcore.py"
        )

    def exec(self, command: str, timeout: float | None = None) -> str:
        raise NotImplementedError

    def upload_file(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError

    def start_agent_process(self, command: str, port: int) -> None:
        raise NotImplementedError

    def get_endpoint(self, port: int) -> tuple[str, dict[str, str]]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def create_agentcore_sandbox(name: str, **kwargs) -> AgentCoreSandbox:
    """Factory function for creating an AgentCoreSandbox."""
    return AgentCoreSandbox(name=name, **kwargs)
