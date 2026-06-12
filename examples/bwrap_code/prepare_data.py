"""Prepare a simple code generation dataset for bwrap sandbox training.

Each task is a simple programming problem with a deterministic expected output.
"""

import json

from rllm.data.dataset import DatasetRegistry

TASKS = [
    {"prompt": "Write a Python program that prints the sum of 3 and 5.", "expected_output": "8"},
    {"prompt": "Write a Python program that prints the product of 7 and 6.", "expected_output": "42"},
    {"prompt": "Write a Python program that prints 'hello world'.", "expected_output": "hello world"},
    {"prompt": "Write a Python program that prints the length of the string 'abcdef'.", "expected_output": "6"},
    {"prompt": "Write a Python program that prints the largest number in [3, 1, 4, 1, 5, 9, 2, 6].", "expected_output": "9"},
    {"prompt": "Write a Python program that prints 2 raised to the power of 10.", "expected_output": "1024"},
    {"prompt": "Write a Python program that prints the first 5 Fibonacci numbers separated by spaces.", "expected_output": "1 1 2 3 5"},
    {"prompt": "Write a Python program that prints the reverse of the string 'python'.", "expected_output": "nohtyp"},
    {"prompt": "Write a Python program that prints the number of vowels in 'hello world'.", "expected_output": "3"},
    {"prompt": "Write a Python program that prints the sum of numbers from 1 to 10.", "expected_output": "55"},
    {"prompt": "Write a Python program that prints the factorial of 5.", "expected_output": "120"},
    {"prompt": "Write a Python program that prints 'even' if 42 is even, else 'odd'.", "expected_output": "even"},
    {"prompt": "Write a Python program that prints the sorted list [5, 2, 8, 1, 9] as space-separated values.", "expected_output": "1 2 5 8 9"},
    {"prompt": "Write a Python program that prints the number of words in 'the quick brown fox'.", "expected_output": "4"},
    {"prompt": "Write a Python program that prints 100 divided by 4 as an integer.", "expected_output": "25"},
    {"prompt": "Write a Python program that prints the absolute value of -17.", "expected_output": "17"},
    {"prompt": "Write a Python program that counts how many numbers from 1 to 20 are divisible by 3 and prints the count.", "expected_output": "6"},
    {"prompt": "Write a Python program that prints the string 'abc' repeated 3 times.", "expected_output": "abcabcabc"},
    {"prompt": "Write a Python program that prints the minimum of 99, 42, and 7.", "expected_output": "7"},
    {"prompt": "Write a Python program that prints the ASCII value of 'A'.", "expected_output": "65"},
    {"prompt": "Write a Python program that prints the square root of 144 as an integer.", "expected_output": "12"},
    {"prompt": "Write a Python program that prints the sum of all even numbers from 1 to 10.", "expected_output": "30"},
    {"prompt": "Write a Python program that prints 'Yes' if 'py' is in 'python', else 'No'.", "expected_output": "Yes"},
    {"prompt": "Write a Python program that prints the binary representation of 10 (without '0b' prefix).", "expected_output": "1010"},
    {"prompt": "Write a Python program that joins ['a', 'b', 'c'] with '-' and prints the result.", "expected_output": "a-b-c"},
    {"prompt": "Write a Python program that prints the floor division of 17 by 3.", "expected_output": "5"},
    {"prompt": "Write a Python program that prints the remainder of 17 divided by 3.", "expected_output": "2"},
    {"prompt": "Write a Python program that prints the number of unique characters in 'mississippi'.", "expected_output": "4"},
    {"prompt": "Write a Python program that prints True if all elements in [2, 4, 6, 8] are even, else False.", "expected_output": "True"},
    {"prompt": "Write a Python program that prints the uppercase version of 'hello'.", "expected_output": "HELLO"},
    {"prompt": "Write a Python program that prints the sum of digits of 12345.", "expected_output": "15"},
    {"prompt": "Write a Python program that prints the GCD of 48 and 18.", "expected_output": "6"},
]


def prepare_bwrap_code_data():
    """Split tasks into train/test and register with DatasetRegistry."""
    test_size = 8
    train_tasks = TASKS[test_size:]
    test_tasks = TASKS[:test_size]

    # Add required fields
    for i, task in enumerate(train_tasks):
        task["data_source"] = "bwrap_code"
        task["question"] = task["prompt"]
    for i, task in enumerate(test_tasks):
        task["data_source"] = "bwrap_code"
        task["question"] = task["prompt"]

    train_dataset = DatasetRegistry.register_dataset(
        "bwrap_code", train_tasks, "train"
    )
    test_dataset = DatasetRegistry.register_dataset(
        "bwrap_code", test_tasks, "test"
    )

    print(f"Train: {len(train_tasks)} tasks, Test: {len(test_tasks)} tasks")
    print(f"Train path: {train_dataset.get_data_path()}")
    print(f"Test path: {test_dataset.get_data_path()}")
    print(f"\nSample: {json.dumps(train_tasks[0], indent=2)}")


if __name__ == "__main__":
    prepare_bwrap_code_data()
