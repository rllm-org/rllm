import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict
import json
import traceback

from rllm.tools.code_tools.code_tool import CodeTool, CodeToolOutput


class PythonInterpreter(CodeTool):
    """A tool for executing Python code in a sandboxed environment."""

    def __init__(self, n_sandboxes=1):
        self.n_workers = n_sandboxes
        self.pool = ProcessPoolExecutor(max_workers=n_sandboxes)
        super().__init__(
            name="local_python",
            description="Execute python code in a local sandbox environment. Returns results and standard output/error.",
            n_sandboxes=n_sandboxes
        )

    @property
    def json(self) -> Dict[str, Any]:
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
                            "description": "Local sandbox to execute the python code in a single cell",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Maximum execution time in seconds before timing out",
                            "default": 12
                        }
                    },
                    "required": ["code"],
                },
            },
        }

    def forward(self, code: str, timeout: int = 12) -> CodeToolOutput:
        """
        Synchronous implementation of Python code execution in a sandbox.
        Uses the process pool for isolation but blocks until completion.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            CodeToolOutput: Contains execution result, stdout, and stderr
        """
        try:
            # Submit the job to the process pool and wait for its result
            future = self.pool.submit(PythonInterpreter._execute_in_subprocess, code, timeout, self.name)

            return future.result(timeout=timeout)
        except Exception as e:
            return CodeToolOutput(
                name=self.name,
                error=f"Sandbox Error: {type(e).__name__} - {str(e)}",
            )

    @staticmethod
    def _check_requirements():
        """Check if required packages are installed and install if missing."""
        required_packages = {
            'sympy': 'sympy',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib'
        }
        
        missing_packages = []
        for package, import_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            try:
                import subprocess
                import sys
                
                # Install missing packages using pip
                subprocess.check_call([
                    sys.executable, 
                    '-m', 'pip', 
                    'install', 
                    '--quiet',
                    *missing_packages
                ])
                print(f"Successfully installed: {', '.join(missing_packages)}")
            except Exception as e:
                raise RuntimeError(f"Failed to install required packages: {str(e)}")

    @staticmethod
    def _execute_in_subprocess(code: str, timeout: int = 10, name: str = "local_python") -> CodeToolOutput:
        """Execute code in a separate process with resource limits."""
        # First check and install requirements
        PythonInterpreter._check_requirements()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap code to capture stdout and stderr, and last expression value
            wrapped_code = f"""
import sys
import io
import contextlib
import math
import json
import traceback

def _format_value(val):
    if val is None:
        return None
    return repr(val)

stdout = io.StringIO()
stderr = io.StringIO()
result = None

with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    try:
        # Split code into lines and get the last line
        code_lines = {repr(code)}.rstrip().split('\\n')
        # Execute all lines except the last one
        if len(code_lines) > 1:
            exec('\\n'.join(code_lines[:-1]))
        # For the last line, try eval first, if it fails, use exec
        try:
            result = eval(code_lines[-1])
        except SyntaxError:
            exec(code_lines[-1])
    except Exception as e:
        stderr.write(traceback.format_exc())

output = {{
    'stdout': stdout.getvalue(),
    'stderr': stderr.getvalue(),
    'result': _format_value(result)
}}
print(json.dumps(output))
"""
            f.write(wrapped_code)
            f.flush()
            
            try:
                # Execute with resource limits
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout,  # Use provided timeout
                )
                try:
                    result_dict = json.loads(result.stdout.strip())
                    return CodeToolOutput(
                        name=name,
                        stdout=result_dict['stdout'],
                        stderr=result_dict['stderr'],
                        output=result_dict['result']
                    )
                except json.JSONDecodeError:
                    return CodeToolOutput(name=name, stderr=f'Error parsing output: {result.stdout}\n{result.stderr}',)
            except subprocess.TimeoutExpired:
                return CodeToolOutput(name=name, stderr=f'Execution timed out after {timeout} seconds',)
            except Exception as e:
                return CodeToolOutput(name=name, error=f"{type(e).__name__} - {str(e)}\n{traceback.format_exc()}",)
            finally:
                os.unlink(f.name)

    def _kill_sandbox(self):
        """Clean up all sandbox resources."""
        if self.pool:
            self.pool.shutdown(wait=True)

    def _init_sandbox(self):
        """Initialize the sandbox environment(s)."""
        if not hasattr(self, 'pool') or self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.n_workers)
            
    def _restart_sandbox(self):
        """Restart the sandbox environment."""
        self._kill_sandbox()
        self._init_sandbox()

if __name__ == "__main__":
    # Create a Python interpreter instance
    interpreter = PythonInterpreter(n_sandboxes=1)

    # Example code to execute
    test_code = """
import math
print('Hello from Python interpreter!')
x = 5
y = 10
print(f'Sum of {x} and {y} is: {x + y}')
for i in range(3):
    print(f"Number: {i}")
x+y-10
"""

    # Run code synchronously
    print("Synchronous result:")
    print(interpreter(code=test_code, use_async=False))
    
    #Run the code using asyncio
    async def test_async_interpreter():
        test_code = """
import math
print('Hello from Python interpreter!')
for i in range(3):
    print(f"Number: {i}")
math.lol
"""
        result = await interpreter(code=test_code, use_async=True)
        print("\nAsynchronous result:")
        print(result)

    # Run the async test
    asyncio.run(test_async_interpreter())
