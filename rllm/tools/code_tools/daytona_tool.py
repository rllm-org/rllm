import os
from typing import Any

try:
    from daytona_sdk import (
        CreateSandboxFromSnapshotParams,
        Daytona,
        DaytonaConfig,
    )
except ImportError:
    Daytona = None
    DaytonaConfig = None
    CreateSandboxFromSnapshotParams = None

from rllm.tools.code_tools.code_tool import CodeTool, CodeToolOutput

DAYTONA_API_KEY = os.environ.get("DAYTONA_API_KEY", None)


class DaytonaPythonInterpreter(CodeTool):
    """Execute Python code in Daytona cloud sandboxes."""

    def __init__(self, n_sandboxes=1, api_key=DAYTONA_API_KEY):
        if Daytona is None:
            raise ImportError(
                "daytona-sdk is not installed. "
                "Install it with `pip install daytona-sdk`."
            )
        assert n_sandboxes > 0, "Number of sandboxes must be greater than 0"
        self.n_sandboxes = n_sandboxes
        self.api_key = api_key
        self.client = Daytona(DaytonaConfig(api_key=self.api_key))
        self._init_sandbox()
        super().__init__(
            name="daytona_python",
            description=(
                "A tool that executes python code in a "
                "Daytona sandbox and returns standard output/error."
            ),
        )

    def _init_sandbox(self):
        """Initialize multiple sandbox environments."""
        self.sandboxes = []
        self.cur_sandbox_idx = 0
        for _ in range(self.n_sandboxes):
            params = CreateSandboxFromSnapshotParams(
                language="python",
            )
            sandbox = self.client.create(params)
            self.sandboxes.append(sandbox)

    def _kill_sandbox(self):
        """Clean up all sandbox resources."""
        for sandbox in self.sandboxes:
            try:
                self.client.delete(sandbox)
            except Exception as e:
                print(f"Error deleting Daytona sandbox: {e}")
        self.sandboxes = []

    def _restart_sandbox(self, id: int = 0) -> Any:
        """Delete and recreate a single sandbox."""
        previous_sandbox = self.sandboxes[id]
        self.client.delete(previous_sandbox)
        params = CreateSandboxFromSnapshotParams(
            language="python",
        )
        sandbox = self.client.create(params)
        self.sandboxes[id] = sandbox
        return sandbox

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

        if id is not None:
            self.cur_sandbox_idx = id % self.n_sandboxes
        else:
            self.cur_sandbox_idx = (self.cur_sandbox_idx + 1) % self.n_sandboxes
        sandbox = self.sandboxes[self.cur_sandbox_idx]

        while max_retries > 0:
            try:
                response = sandbox.process.code_run(code, timeout=timeout)
                break
            except Exception:
                max_retries -= 1
                if max_retries == 0:
                    self._restart_sandbox(self.cur_sandbox_idx)
                    return CodeToolOutput(
                        name=self.name or "daytona_python",
                        error="Sandbox error, please try again.",
                    )

        stdout = response.result if response.result else None
        stderr = None
        output = None

        # exit_code != 0 means the code errored — stdout contains the traceback
        if response.exit_code != 0:
            stderr = response.result
            stdout = None

        return CodeToolOutput(
            name=self.name or "daytona_python",
            stdout=stdout,
            stderr=stderr,
            output=output,
        )

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
                            "description": (
                                "Python code to execute in a "
                                "Daytona sandbox environment."
                            ),
                        }
                    },
                    "required": ["code"],
                },
            },
        }


if __name__ == "__main__":
    from pprint import pprint

    interpreter = DaytonaPythonInterpreter()
    pprint(
        interpreter(
            "print('Hello, world!')\nprint('Run run run.')\n"
            "import math\nmath.sqrt(4)\nmath.sqrt(3)"
        )
    )

    # Run the code using asyncio
    import asyncio

    async def test_interpreter():
        coro = interpreter(
            code="print('Hello, world!')\nimport math\n"
            "math.sqrt(4)\nmath.sqrt(3)\nmath.lol",
            use_async=True,
        )
        print("Starting coroutine...")
        result = await coro
        pprint(result)

    asyncio.run(test_interpreter())
