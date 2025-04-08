import asyncio
import json
import inspect
import typing
import os

from rllm.rewards.code_utils.utils import TOGETHER_IMPORTS

def chat_completion_with_tool(
    client: "AsyncOpenAI",
    tool_caller: "ToolCaller",
    messages_list,
    model="gpt-4",
    max_round=20,
    batch_size=32,  # Added batch_size parameter
):
    from openai import AsyncOpenAI

    async def apply_tool(completion, messages, tool_caller, id=None):
        tool_calls = tool_caller.parse_tool_calls(completion.choices[0].message.content)

        if len(tool_calls) > 0:
            tool_call = tool_calls[0]
            if id is not None and isinstance(tool_call["parameters"], dict):
                tool_call["parameters"]["id"] = id
            tool_call_result = await tool_caller(tool_call["name"], tool_call["parameters"])
            print("tool_call_result", tool_call_result)
            messages.append(tool_call_result)
            return True

        return False

    async def tool_call_flow(example, request_id):
        try:
            messages = example["messages"]
            tool_infos = tool_caller.get_tool_infos()

            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_infos,
                temperature=0.6,
                max_tokens=8192,
                top_p=0.95,
                stop=["```\n\n"],
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": completion.choices[0].message.content + "```\n\n",
                }
            )
            print("round: 0", completion.choices[0].message.content)
            curr_round = 0
            while curr_round < max_round:
                use_tools = await apply_tool(
                    completion, messages, tool_caller, id=request_id
                )
                if use_tools:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tool_infos,
                        temperature=0.6,
                        max_tokens=8192,
                        top_p=0.95,
                        stop=["```\n\n"],
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": completion.choices[0].message.content
                            + "```\n\n",
                        }
                    )
                else:
                    break

                curr_round += 1
                print(f"round {curr_round}:", completion.choices[0].message.content)
        except Exception as e:
            print("Exception:", str(e))
            pass

        return example

    async def run_batch():
        # Initialize pool with first batch of requests
        active_requests = []
        results = []
        messages_iter = iter(messages_list)
        processed_count = 0
        request_id = 0

        # Fill initial pool
        for _ in range(batch_size):
            try:
                messages = next(messages_iter)
                task = asyncio.create_task(tool_call_flow(messages, request_id))
                active_requests.append((task, request_id))
                request_id += 1
            except StopIteration:
                break

        # Process requests and refill pool
        while active_requests:
            done, pending = await asyncio.wait(
                [task for task, _ in active_requests],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Update active_requests with pending tasks
            active_requests = [
                (task, id) for task, id in active_requests if task in pending
            ]

            for completed_task in done:
                # Find the ID for the completed task
                # task_id = next(id for task, id in active_requests if task == completed_task)
                final_messages = await completed_task
                # results.append({"id": task_id, "data": final_messages})
                results.append(final_messages)
                processed_count += 1

                # Save results checkpoint every 200 examples
                if processed_count % 600 == 0:
                    with open("messages_checkpoint.json", "w") as f:
                        json.dump(results, f, indent=2)

                # Try to add new request to maintain pool
                try:
                    messages = next(messages_iter)
                    new_task = asyncio.create_task(tool_call_flow(messages, request_id))
                    active_requests.append((new_task, request_id))
                    request_id += 1
                except StopIteration:
                    pass

            print("Active requests:", len(active_requests))

        return results

    return asyncio.run(run_batch())

def function_to_dict(func):
    """
    Converts a function into a dictionary representation suitable for JSON Schema.

    Parameters:
        func (function): The function to convert.

    Returns:
        dict: A dictionary representing the function in JSON Schema format.
    """
    # Get the function name
    func_name = func.__name__

    # Get the docstring
    docstring = func.__doc__ or ''

    # Get the function signature
    sig = inspect.signature(func)

    # Initialize the parameters dictionary
    params = {
        "type": "object",
        "properties": {},
        "required": []
    }

    # Map Python types to JSON Schema types
    type_mapping = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        dict: "object",
        list: "array",
    }

    for param_name, param in sig.parameters.items():
        # Get the type annotation
        annotation = param.annotation

        param_type = "string"  # Default type
        param_description = ""

        # Determine the JSON Schema type and description
        if annotation != inspect.Parameter.empty:
            # Handle Annotated types
            origin = typing.get_origin(annotation)
            if origin is typing.Annotated:
                args = typing.get_args(annotation)
                base_type = args[0]
                metadata = args[1:]
                param_type = type_mapping.get(base_type, "string")
                # Assuming the first metadata argument is the description
                if metadata:
                    param_description = metadata[0]
            else:
                param_type = type_mapping.get(annotation, "string")
        # Add the parameter to properties
        param_schema = {"type": param_type}
        if param_description:
            param_schema["description"] = param_description
        params["properties"][param_name] = param_schema

        # Add to required if there's no default value
        if param.default == inspect.Parameter.empty:
            params["required"].append(param_name)

    # Build the final dictionary
    function_dict = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": docstring.strip().split('\n')[0],  # First line of docstring
            "parameters": params
        }
    }

    return function_dict

def _extract_import_lines(code: str) -> list:
    import_lines = []
    others = []
    lines = code.splitlines()
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(line)
        else:
            others.append(line)
    return "\n".join(import_lines), "\n".join(others)

def stdin_test_code_wrapper(code: str, tests) -> str:
    """
    Wraps the input code with a function definition and a call to that function.
    This is useful for testing purposes, especially when using tools like TogetherCodeTool.

    Args:
        code (str): The code to be wrapped.

    Returns:
        str: The wrapped code.
    """
    # separate import lines from the rest of the code
    imports, others = _extract_import_lines(code)

    indented_code = others.replace('\n', '\n    ')

    wrapped_code = f"""
import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
{TOGETHER_IMPORTS}
{imports}

class StringIOWithBuffer(io.StringIO):
    def __init__(self, initial_value="", newline="\\n"):
        super().__init__(initial_value, newline)
        # Create an underlying bytes buffer initialized with the encoded input.
        self._buffer = io.BytesIO(initial_value.encode("utf-8"))
    
    def write(self, s):
        super().write(s)
        # Update the underlying buffer as well (if needed)
        self._buffer.write(s.encode("utf-8"))
    
    @property
    def buffer(self):
        return self._buffer

# Save original stdin and stderr
original_stdin = sys.stdin
original_stderr = sys.stderr

# Function to run tests with the original code
def run_tests():
    # Original code
    {indented_code}

# Function to exit with error after cleanup
def exit_with_error():
    # Restore original stdin and stderr before exiting
    sys.stdin = original_stdin
    sys.stderr = original_stderr
    sys.exit(1)

# Test with provided stdin inputs
stdin_test_cases = {tests}

for i, test_case in enumerate(stdin_test_cases):
    # Prepare stdin simulation
    if isinstance(test_case["input"], list):
        test_case["input"] = "\\n".join(test_case["input"]) + "\\n"
        test_case["output"] = "\\n".join(test_case["output"]) + "\\n"
    sys.stdin = StringIOWithBuffer(test_case["input"])
    input = lambda : sys.stdin.readline().strip()
    
    # Capture stdout and stderr
    captured_output = StringIOWithBuffer()
    captured_stderr = io.StringIO()
    
    with redirect_stdout(captured_output), redirect_stderr(captured_stderr):
        try:
            # Run the code in a fresh environment for each test
            run_tests()
        except Exception as e:
            stderr_output = captured_stderr.getvalue()
            if stderr_output:
                original_stderr.write(f"Stderr from test case {{i+1}}:\\n{{stderr_output}}\\n")
            
            print(f"Error in test case {{i+1}}: {{str(e)}}")
            print(f"Input: {{test_case['input']}}")
            exit_with_error()
    
    # Get the captured output
    actual_output = captured_output.getvalue().strip()
    expected_output = test_case["output"].strip()
    stderr_output = captured_stderr.getvalue().strip()
    
    # Write stderr output to the outer code's stderr
    if stderr_output:
        original_stderr.write(f"Stderr from test case {{i+1}}:\\n{{stderr_output}}\\n")
        original_stderr.flush()
    
    # Print test results
    print("Result: " + ("PASS" if actual_output == expected_output else "FAIL"))
    
    # If test fails, exit immediately
    if actual_output != expected_output:
        print("Test failed. Exiting.")
        exit_with_error()

# Restore original stdin and stderr
sys.stdin = original_stdin
sys.stderr = original_stderr

# If we got here, all tests passed
print("All tests passed successfully!")
"""

    return wrapped_code

def call_based_test_code_wrapper(code: str, tests) -> str:
    """
    Wraps the input code with a test harness that calls a given function using provided input arguments,
    then compares the function's return value with expected outputs.
    
    Args:
        code (str): The student's code as a string.
        tests (dict): A dictionary containing:
            - 'fn_name': The name of the function to call.
            - 'inputs': A list of lists. Each inner list contains the positional arguments for a test case.
            - 'outputs': A list of lists. Each inner list contains the expected result(s).
    
    Returns:
        str: A complete self-contained Python script as a string.
    """
    wrapped_code = f"""#!/usr/bin/env python3
{code}

import sys

def run_tests():
    # Test data (call-based)
    test_data = {tests}
    fn_name = test_data["fn_name"]
    inputs_list = test_data["inputs"]
    outputs_list = test_data["outputs"]

    # Retrieve the function from the global namespace
    try:
        fn = globals()[fn_name]
    except KeyError:
        print(f"Function '{{fn_name}}' is not defined.")
        sys.exit(1)
    
    all_passed = True
    for i, (inp, expected) in enumerate(zip(inputs_list, outputs_list)):
        try:
            result = fn(*inp)
        except Exception as e:
            print(f"Test case {{i+1}} raised an exception: {{e}}")
            all_passed = False
            continue
        
        # Here we assume expected is given as a list where the first element is the expected result.
        if result != expected[0]:
            print(f"Test case {{i+1}} failed:")
            print("Input:", inp)
            print("Expected:", expected[0])
            print("Got:", result)
            all_passed = False
        else:
            print(f"Test case {{i+1}} passed.")
    
    if not all_passed:
        sys.exit(1)
    else:
        print("All tests passed successfully!")

if __name__ == "__main__":
    run_tests()
"""
    return wrapped_code
