import ast
import faulthandler
import multiprocessing
import queue
import signal
import traceback


from rllm.tools.code_tools.code_tool import CodeTool, CodeToolOutput
from rllm.rewards.code_utils.livecodebench import (
    Capturing,
    clean_if_name,
    compile_code,
    get_function,
    make_function,
    reliability_guard,
    timeout_handler,
)


def lcb_sandbox(code, timeout):
    """
    Execute Python code in a sandboxed environment with timeout protection.
    
    This function runs the provided code in a separate process with safety measures
    to prevent harmful operations and ensure termination after the specified timeout.
    
    Args:
        code (str): Python code to execute
        timeout (int): Maximum execution time in seconds
        
    Returns:
        tuple: (stdout, stderr, result) containing the execution output and result
    """
    def ensure_return_value(code):
        """
        Ensures the code has a return statement for the last expression.
        Only converts the last statement to a return statement if it's an expression.
        
        Args:
            code (str): Python code to process
            
        Returns:
            str: Modified code with return statement if needed
        """
        if not code.strip():
            return code
        
        try:
            # Parse the code
            tree = ast.parse(code)
            body = tree.body
            
            # If the last element is an expression, convert it to a return statement
            if body and isinstance(body[-1], ast.Expr):
                value = body[-1].value
                body[-1] = ast.Return(value=value)
                
                # Preserve the line numbers and column offsets for better error messages
                ast.fix_missing_locations(tree)
                
            # Unparse the modified AST back to code
            return ast.unparse(tree)
        except SyntaxError:
            # If the code has syntax errors, return the original code
            return code
        except Exception as e:
            # Log other unexpected errors but return the original code
            print(f"Warning: Could not process code: {e}")
            return code
    
    # Preprocess the code to ensure the last expression is returned
    code = ensure_return_value(code)
    
    def execute_code(code, timeout):
        """
        Execute the provided code with safety measures and timeout handling.
        
        Args:
            code (str): Python code to execute
            timeout (int): Maximum execution time in seconds
            
        Returns:
            tuple: (stdout, stderr, result) containing execution output and result
        """
        signal.signal(signal.SIGALRM, timeout_handler)
        stdout, stderr, result = None, None, None
        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()
        signal.alarm(timeout)
        try:
            code = clean_if_name(code)
            ## we wrap the given code inside another function
            code = make_function(code)
            compiled_sol = compile_code(code, timeout)
            if compiled_sol is None:
                stderr = "Failed to compile code"
                return stdout, stderr, result
            method = get_function(compiled_sol, "wrapped_function")
            if method is None:
                stderr = "Failed to get function 'wrapped_function'"
                return stdout, stderr, result
            signal.alarm(timeout)
            faulthandler.enable()
            signal.alarm(timeout)
            with Capturing() as captured_output:
                try:
                    try:
                        result = method()
                    except SystemExit as e:
                        stderr = f"SystemExit: {e}"
                    finally:
                        pass
                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    if "timeoutexception" in repr(e).lower():
                        stderr = f"Time Limit Exceeded."
                    else:
                        stderr = traceback.format_exc()
                finally:
                    signal.alarm(0)
                    faulthandler.disable()
            stdout = captured_output[0] if captured_output else ""
            return stdout, stderr, result
        except Exception as e:
            return stdout, stderr, result
        finally:
            signal.alarm(0)


    # Use multiprocessing to isolate code execution in a separate process
    manager = multiprocessing.Manager()
    result = manager.Queue()

    def _wrapper_exec_fn(sample, timeout):
        """Helper function to execute code and put results in the queue"""
        res = execute_code(sample, timeout=timeout)
        result.put(res)

    # Create and start the process
    p = multiprocessing.Process(
        target=_wrapper_exec_fn,
        args=(code, timeout),
    )
    p.start()
    
    # Wait for the process to complete with additional buffer time
    p.join(
        timeout=(timeout + 1) + 5
    )
    
    try: 
        # Get the result from the queue
        res = result.get()
        return res
    except queue.Empty:
        # Return timeout message if no result is available
        return "Timeout"
    finally:
        # Ensure the process is terminated if still running
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
            if p.is_alive():
                p.kill()


class LCBPythonInterpreter(CodeTool):
    """
    A tool for executing Python code in a sandboxed environment.
    
    This tool provides a safe way to execute Python code with timeout protection
    and isolation from the main process, using the LiveCodeBench execution environment.
    """

    def __init__(self):
        """Initialize the Python interpreter tool with appropriate settings."""
        super().__init__(
            name="python",
            description="Execute python code in the same environment as the LiveCodeBench benchmark.",
            n_sandboxes=-1,
        )

    def forward(self, code: str, timeout: int = 12) -> CodeToolOutput:
        """
        Execute Python code using the LiveCodeBench sandbox environment.
        
        Args:
            code (str): Python code to execute
            timeout (int): Maximum execution time in seconds, defaults to 12
            
        Returns:
            CodeToolOutput: Contains execution results with stdout, stderr, and result fields
        """
        try:
            stdout, stderr, result = lcb_sandbox(code, timeout=timeout)
            return CodeToolOutput(
                name=self.name,
                stdout=stdout,
                stderr=stderr,
                output=result
            )
        except Exception as e:
            return CodeToolOutput(
                name=self.name,
                error=f"Sandbox Error: {type(e).__name__} - {str(e)}",
            )


if __name__ == "__main__":
    # Create a Python interpreter instance
    interpreter = LCBPythonInterpreter()

    # Example code to execute
    test_code = """
# Generate a large amount of code
result = 0
for i in range(1000):
    exec(f"var_{i} = {i}")
    result += i

# Final expression after lots of code
result  # Should be converted to return
"""

    # Run code
    print(interpreter(code=test_code))
