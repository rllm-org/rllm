"""Modal sandbox backend (stub).

Install the ``modal`` package and implement this backend to run
agent code in Modal serverless sandboxes.
"""

from __future__ import annotations


class ModalSandbox:
    """Stub for Modal sandbox backend."""

    def __init__(self, name: str, **kwargs):
        raise NotImplementedError(
            "Modal sandbox backend is not yet implemented. "
            "To use Modal sandboxes, install the `modal` package and "
            "implement ModalSandbox in rllm/sdk/sandbox/backends/modal_backend.py"
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


def create_modal_sandbox(name: str, **kwargs) -> ModalSandbox:
    """Factory function for creating a ModalSandbox."""
    return ModalSandbox(name=name, **kwargs)
