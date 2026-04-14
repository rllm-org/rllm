#!/usr/bin/env python3
import asyncio

import pytest

from rllm.tools import tool_registry
from rllm.tools.tool_base import ToolOutput

# Test code for Python interpreter
python_test_cases = [
    {
        "name": "Basic Arithmetic",
        "code": """
print("Testing basic arithmetic...")
x = 10
y = 5
print(f"Addition: {x + y}")
print(f"Subtraction: {x - y}")
print(f"Multiplication: {x * y}")
print(f"Division: {x / y}")
print(f"Integer Division: {x // y}")
print(f"Modulo: {x % y}")
print(f"Power: {x ** y}")
""",
        "expected_stdout": "Testing basic arithmetic...\nAddition: 15\nSubtraction: 5\nMultiplication: 50\nDivision: 2.0\nInteger Division: 2\nModulo: 0\nPower: 100000",
    },
    {
        "name": "List Operations",
        "code": """
print("Testing list operations...")
numbers = [1, 2, 3, 4, 5]
print(f"Original list: {numbers}")
print(f"Sum: {sum(numbers)}")
print(f"Average: {sum(numbers)/len(numbers)}")
print(f"Max: {max(numbers)}")
print(f"Min: {min(numbers)}")
squared = [x**2 for x in numbers]
print(f"Squared: {squared}")
""",
        "expected_stdout": "Testing list operations...\nOriginal list: [1, 2, 3, 4, 5]\nSum: 15\nAverage: 3.0\nMax: 5\nMin: 1\nSquared: [1, 4, 9, 16, 25]",
    },
    {
        "name": "Error Handling",
        "code": """
print("Testing error handling...")
try:
    result = 1/0
except ZeroDivisionError as e:
    print(f"Caught error: {e}")
try:
    undefined_var
except NameError as e:
    print(f"Caught error: {e}")
""",
        "expected_stdout": "Testing error handling...\nCaught error: division by zero\nCaught error: name 'undefined_var' is not defined",
    },
    {
        "name": "File Operations",
        "code": """
print("Testing file operations...")
test_file = "test_output.txt"
with open(test_file, "w") as f:
    f.write("Hello, World!")
with open(test_file, "r") as f:
    content = f.read()
print(f"File content: {content}")
print("File operations completed successfully")
""",
        "expected_stdout": "Testing file operations...\nFile content: Hello, World!\nFile operations completed successfully",
    },
]

# Test code for async Python interpreter
python_async_test_cases = [
    {
        "name": "Basic Async",
        "code": """
import asyncio
import time

async def count():
    for i in range(3):
        print(f"Count: {i}")
        await asyncio.sleep(0.1)

print("Starting async test...")
asyncio.run(count())
print("Async test complete!")
""",
        "expected_stdout": "Starting async test...\nCount: 0\nCount: 1\nCount: 2\nAsync test complete!",
    },
    {
        "name": "Multiple Async Tasks",
        "code": """
import asyncio
import time

async def task(name, delay):
    print(f"Task {name} started")
    await asyncio.sleep(delay)
    print(f"Task {name} completed")

async def main():
    print("Starting multiple tasks...")
    tasks = [
        task("A", 0.1),
        task("B", 0.2),
        task("C", 0.3)
    ]
    await asyncio.gather(*tasks)
    print("All tasks completed!")

asyncio.run(main())
""",
        "expected_stdout": "Starting multiple tasks...\nTask A started\nTask B started\nTask C started\nTask A completed\nTask B completed\nTask C completed\nAll tasks completed!",
    },
]

# Test queries for search tools
search_test_cases = {
    "google_search": [
        {"name": "Basic Search", "query": "What is Python programming?", "expected_fields": ["title", "snippet", "link"]},
        {"name": "Technical Search", "query": "Python async await syntax example", "expected_fields": ["title", "snippet", "link"]},
    ],
    # "tavily_search": [
    #     {"name": "News Search", "query": "Latest developments in AI", "expected_fields": ["title", "snippet", "url"]},
    #     {"name": "Technical Search", "query": "Python type hints tutorial", "expected_fields": ["title", "snippet", "url"]},
    # ],
    # "tavily_extract": [
    #     {"name": "Python.org", "url": "https://www.python.org/about/", "expected_fields": ["title", "text"]},
    #     {"name": "Python Docs", "url": "https://docs.python.org/3/tutorial/", "expected_fields": ["title", "text"]},
    # ],
    # "firecrawl": [
    #     {"name": "Python.org", "url": "https://www.python.org", "expected_fields": ["title", "text", "links"]},
    #     {"name": "Python Docs", "url": "https://docs.python.org/3/", "expected_fields": ["title", "text", "links"]},
    # ],
}


def validate_tool_output(result: ToolOutput, expected_fields: list | None = None) -> bool:
    """Validate the tool output has the expected structure and content."""
    if result.error:
        print(f"Error in tool execution: {result.error}")
        return False

    if expected_fields:
        if isinstance(result.output, dict):
            missing_fields = [field for field in expected_fields if field not in result.output]
            if missing_fields:
                print(f"Missing expected fields: {missing_fields}")
                return False
        else:
            print("Expected dictionary output with fields")
            return False

    return True


def test_python_tool():
    """Test the Python interpreter tool synchronously and asynchronously."""
    print("\nTesting Python interpreter tool...")

    # Get the tool using the instantiate method
    python_tool = tool_registry.instantiate("python")

    # Test sync cases
    print("\nTesting sync execution:")
    for test_case in python_test_cases:
        print(f"\nRunning test: {test_case['name']}")
        print("Code:")
        print(test_case["code"])

        result = python_tool(test_case["code"])
        print("Result:")
        print(f"Error: {result.error}")
        print(f"Output: {result.output}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # Handle None values properly
        actual_stdout = result.stdout.strip() if result.stdout else ""
        expected_stdout = test_case["expected_stdout"].strip()

        if result.error:
            print(f"Test failed with error: {result.error}")
        elif actual_stdout == expected_stdout:
            print("Test passed!")
        else:
            print("Test failed! Output doesn't match expected")
            print("Expected:")
            print(expected_stdout)
            print("Actual:")
            print(actual_stdout)

    # Test async cases
    print("\nTesting async execution:")
    for test_case in python_async_test_cases:
        print(f"\nRunning test: {test_case['name']}")
        print("Code:")
        print(test_case["code"])

        result = asyncio.run(python_tool(test_case["code"], use_async=True))
        print("Result:")
        print(f"Error: {result.error}")
        print(f"Output: {result.output}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # Handle None values properly
        actual_stdout = result.stdout.strip() if result.stdout else ""
        expected_stdout = test_case["expected_stdout"].strip()

        if result.error:
            print(f"Test failed with error: {result.error}")
        elif actual_stdout == expected_stdout:
            print("Test passed!")
        else:
            print("Test failed! Output doesn't match expected")
            print("Expected:")
            print(expected_stdout)
            print("Actual:")
            print(actual_stdout)


@pytest.mark.parametrize("tool_name", search_test_cases.keys())
def test_search_tool(tool_name: str):
    """Test a search tool with multiple test cases."""
    print(f"\nTesting {tool_name} tool...")

    # Get the tool using the instantiate method
    tool = tool_registry.instantiate(tool_name)

    # Run test cases
    for test_case in search_test_cases[tool_name]:
        print(f"\nRunning test: {test_case['name']}")
        print(f"Query/URL: {test_case.get('query', test_case.get('url'))}")

        # Execute tool
        if "query" in test_case:
            result = tool(test_case["query"])
        else:
            result = tool(test_case["url"])

        print("Result:")
        print(f"Error: {result.error}")
        print(f"Output: {result.output}")

        # Validate output
        if validate_tool_output(result, test_case["expected_fields"]):
            print("Test passed!")
        else:
            print("Test failed!")
