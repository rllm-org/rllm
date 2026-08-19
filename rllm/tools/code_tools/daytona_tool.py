import os
from typing import Any

try:
    from daytona import (
        CreateSandboxFromSnapshotParams,
        Daytona,
        DaytonaAuthenticationError,
        DaytonaConfig,
    )
except ImportError:
    CreateSandboxFromSnapshotParams = None
    Daytona = None
    DaytonaAuthenticationError = None
    DaytonaConfig = None

from rllm.tools.code_tools.code_tool import CodeTool, CodeToolOutput

DAYTONA_API_KEY = os.environ.get("DAYTONA_API_KEY", None)


class DaytonaPythonInterpreter(CodeTool):
    """Execute Python code in a Daytona sandbox"""

    def __init__(
        self,
        n_sandboxes: int = 1,
        api_key: str | None = DAYTONA_API_KEY,
        api_url: str | None = None,
        snapshot: str | None = None,
        env_vars: dict[str, str] | None = None,
    ):
        if Daytona is None:
            raise ImportError("daytona is not installed. Please install it with `pip install daytona`.")
        assert n_sandboxes > 0, "Number of sandboxes must be greater than 0"

        super().__init__(
            name="daytona_python",
            description="A tool that executes python code in a Daytona sandbox and returns standard output/error.",
            n_sandboxes=n_sandboxes,
        )

        self.api_key = api_key
        self.api_url = api_url
        self.snapshot = snapshot
        self.env_vars = env_vars
        self._init_sandbox()

    def _init_sandbox(self):
        """Initialize multiple sandbox environments."""
        config_kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            config_kwargs["api_key"] = self.api_key
        if self.api_url is not None:
            config_kwargs["api_url"] = self.api_url
        self._client = Daytona(DaytonaConfig(**config_kwargs))

        params_kwargs: dict[str, Any] = {"language": "python"}
        if self.snapshot is not None:
            params_kwargs["snapshot"] = self.snapshot
        if self.env_vars is not None:
            params_kwargs["env_vars"] = dict(self.env_vars)
        params = CreateSandboxFromSnapshotParams(**params_kwargs)

        self.sandboxes = []
        self.cur_sandbox_idx = 0
        for _ in range(self.n_sandboxes):
            self.sandboxes.append(self._client.create(params))

    def _kill_sandbox(self):
        """Clean up all sandbox resources."""
        for sandbox in getattr(self, "sandboxes", []):
            try:
                sandbox.delete()
            except Exception as e:
                print(f"Error deleting sandbox: {e}")
        self.sandboxes = []

    def forward(self, code: str, timeout: int = 20, **kwargs) -> CodeToolOutput:
        """
        Execute Python code in one of the sandboxes using round-robin distribution.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            **kwargs: Additional parameters including id, max_retries

        Returns:
            CodeToolOutput containing execution results, stdout, and stderr
        """
        id = kwargs.get("id", None)
        max_retries = kwargs.get("max_retries", 3)

        if id:
            self.cur_sandbox_idx = id % self.n_sandboxes
        else:
            self.cur_sandbox_idx = (self.cur_sandbox_idx + 1) % self.n_sandboxes
        sandbox = self.sandboxes[self.cur_sandbox_idx]

        for _ in range(max_retries):
            try:
                response = sandbox.process.code_run(code, timeout=timeout)
                break
            except DaytonaAuthenticationError as e:
                return CodeToolOutput(name=self.name or "daytona_python", error=f"Auth error: {e}")
            except Exception:
                continue
        else:
            return CodeToolOutput(name=self.name or "daytona_python", error="Sandbox error, please try again.")

        stdout = None
        stderr = None
        error = None
        if response.exit_code == 0:
            stdout = response.result or None
        else:
            stderr = response.result or None
            if response.result:
                lines = response.result.strip().splitlines()
                if lines:
                    error = lines[-1]

        return CodeToolOutput(name=self.name or "daytona_python", stdout=stdout, stderr=stderr, error=error)

    @property
    def json(self) -> dict[str, Any]:
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
                            "description": "Python code to execute in a Daytona sandbox environment.",
                        },
                    },
                    "required": ["code"],
                },
            },
        }


if __name__ == "__main__":
    from pprint import pprint

    interpreter = DaytonaPythonInterpreter()
    try:
        pprint(interpreter(code="print('Hello, world!')\nimport math\nmath.sqrt(4)"))
        pprint(interpreter(code="1/0"))
    finally:
        interpreter._kill_sandbox()
