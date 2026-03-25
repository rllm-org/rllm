from typing import Any, Literal

from rllm.tools.code_tools.code_tool import CodeTool, CodeToolOutput
from rllm.tools.code_tools.daytona_tool import DaytonaPythonInterpreter
from rllm.tools.code_tools.e2b_tool import E2BPythonInterpreter
from rllm.tools.code_tools.lcb_tool import LCBPythonInterpreter
from rllm.tools.code_tools.together_tool import TogetherCodeTool

# Backend types
BackendType = Literal["local", "e2b", "together", "lcb", "daytona"]


class PythonInterpreter(CodeTool):
    """
    A unified Python interpreter tool that supports multiple backends.

    This class provides a common interface for executing Python code using different
    backend implementations, including local execution, E2B sandbox, Together API,
    and LiveCodeBench environment.
    """

    def __init__(
        self,
        backend: BackendType = "local",
        n_sandboxes: int = 1,
        api_key: str | None = None,
        name: str = "python",
        description: str = (
            "Execute Python code in a sandboxed environment."
            " Returns results and standard output/error."
        ),
    ):
        """
        Initialize the unified Python interpreter with the specified backend.

        Args:
            backend: The backend to use
                ("local", "e2b", "together", "lcb", or "daytona")
            n_sandboxes: Number of concurrent sandboxes/workers
                to use (for applicable backends)
            api_key: API key for cloud-based backends
                (e2b, together, daytona)
            name: The name of the tool
            description: Description of what the tool does
        """
        self.backend_type = backend
        self.n_sandboxes = n_sandboxes
        self.api_key = api_key

        # Initialize the appropriate backend
        self._init_backend()

        super().__init__(name=name, description=description, n_sandboxes=n_sandboxes)

    def _init_backend(self):
        """Initialize the selected backend interpreter."""
        if self.backend_type == "local":
            self.backend: (
                LCBPythonInterpreter
                | E2BPythonInterpreter
                | TogetherCodeTool
                | DaytonaPythonInterpreter
            ) = LCBPythonInterpreter()
        elif self.backend_type == "e2b":
            self.backend = E2BPythonInterpreter(
                n_sandboxes=self.n_sandboxes, api_key=self.api_key
            )
        elif self.backend_type == "together":
            self.backend = TogetherCodeTool(api_key=self.api_key)
        elif self.backend_type == "daytona":
            self.backend = DaytonaPythonInterpreter(
                n_sandboxes=self.n_sandboxes, api_key=self.api_key
            )
        else:
            raise ValueError(f"Unsupported backend type: {self.backend_type}")

    def forward(self, code: str, timeout: int = 12, **kwargs) -> CodeToolOutput:
        """
        Execute Python code using the selected backend.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            **kwargs: Additional parameters specific to the backend implementation

        Returns:
            CodeToolOutput containing execution results, stdout, and stderr
        """
        return self.backend.forward(code=code, timeout=timeout, **kwargs)

    def _init_sandbox(self):
        """Initialize the sandbox environment."""
        if hasattr(self, "backend") and hasattr(self.backend, "_init_sandbox"):
            self.backend._init_sandbox()

    def _kill_sandbox(self):
        """Clean up all sandbox resources."""
        if hasattr(self, "backend") and hasattr(self.backend, "_kill_sandbox"):
            self.backend._kill_sandbox()

    def _restart_sandbox(self):
        """Restart the sandbox environment."""
        if hasattr(self, "backend") and hasattr(self.backend, "_restart_sandbox"):
            self.backend._restart_sandbox()
        else:
            self._kill_sandbox()
            self._init_backend()

    @property
    def json(self) -> dict[str, Any]:
        """Return the tool's information in the required format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Execute Python code in a sandboxed"
                                " environment. Returns results and"
                                " standard output/error."
                            ),
                        },
                        "timeout": {
                            "type": "integer",
                            "description": (
                                "Maximum execution time in seconds before timing out"
                            ),
                            "default": 12,
                        },
                    },
                    "required": ["code"],
                },
            },
        }
