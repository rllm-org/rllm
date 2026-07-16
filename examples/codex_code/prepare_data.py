"""Prepare medium-difficulty coding tasks for Codex CLI RL training.

Tasks are harder than sandbox_code's trivial arithmetic: algorithms,
multi-step logic, string manipulation, and file I/O.  Target pass
rate with a 9B model: 30-70%.

Each task has:
- prompt: the coding problem description
- expected_output: exact stdout to match
- setup_code: optional shell commands to create input files before the agent runs
"""

from rllm.data.dataset import DatasetRegistry

TASKS = [
    # --- String manipulation ---
    {
        "prompt": "Write a Python program that reads the string 'racecar' and prints 'palindrome' if it's a palindrome, else 'not palindrome'.",
        "expected_output": "palindrome",
    },
    {
        "prompt": "Write a Python program that prints the most frequent character in 'abracadabra'. If there's a tie, print the one that comes first alphabetically.",
        "expected_output": "a",
    },
    {
        "prompt": "Write a Python program that takes the string 'hello world from python' and prints each word capitalized, separated by spaces.",
        "expected_output": "Hello World From Python",
    },
    {
        "prompt": "Write a Python program that prints the longest word in the sentence 'the quick brown fox jumps over the lazy dog'.",
        "expected_output": "quick",
    },
    {
        "prompt": "Write a Python program that removes all duplicate characters from 'programming' keeping only the first occurrence, and prints the result.",
        "expected_output": "progamin",
    },
    {
        "prompt": "Write a Python program that prints the string 'hello world' with each word reversed but word order preserved.",
        "expected_output": "olleh dlrow",
    },
    {
        "prompt": "Write a Python program that counts the number of consonants in 'synchronize' and prints the count.",
        "expected_output": "7",
    },
    {
        "prompt": "Write a Python program that compresses the string 'aaabbbccddddee' using run-length encoding and prints the result (e.g., 'a3b3c2d4e2').",
        "expected_output": "a3b3c2d4e2",
    },
    # --- Algorithms ---
    {
        "prompt": "Write a Python program that finds the second largest number in [45, 12, 78, 34, 56, 78, 90, 23] and prints it.",
        "expected_output": "78",
    },
    {
        "prompt": "Write a Python program that checks if 197 is prime and prints 'prime' or 'not prime'.",
        "expected_output": "prime",
    },
    {
        "prompt": "Write a Python program that prints the LCM (least common multiple) of 12 and 18.",
        "expected_output": "36",
    },
    {
        "prompt": "Write a Python program that prints the first 10 prime numbers separated by spaces.",
        "expected_output": "2 3 5 7 11 13 17 19 23 29",
    },
    {
        "prompt": "Write a Python program that converts the decimal number 255 to hexadecimal (lowercase, without '0x' prefix) and prints it.",
        "expected_output": "ff",
    },
    {
        "prompt": "Write a Python program that prints the number of ways to make change for 10 cents using coins of 1, 5, and 10 cents.",
        "expected_output": "4",
    },
    {
        "prompt": "Write a Python program that implements binary search to find 42 in [1, 5, 12, 22, 33, 42, 55, 67, 89] and prints its index.",
        "expected_output": "5",
    },
    {
        "prompt": "Write a Python program that computes the edit distance between 'kitten' and 'sitting' and prints it.",
        "expected_output": "3",
    },
    # --- Multi-step logic ---
    {
        "prompt": "Write a Python program that generates a 3x3 identity matrix and prints it row by row, with elements separated by spaces.",
        "expected_output": "1 0 0\n0 1 0\n0 0 1",
    },
    {
        "prompt": "Write a Python program that flattens the nested list [[1, 2], [3, [4, 5]], [6]] into a single list and prints the elements separated by spaces.",
        "expected_output": "1 2 3 4 5 6",
    },
    {
        "prompt": "Write a Python program that implements the Caesar cipher with a shift of 3 on 'hello' and prints the encrypted text.",
        "expected_output": "khoor",
    },
    {
        "prompt": "Write a Python program that prints the first 8 terms of the sequence: 0, 1, 1, 2, 4, 7, 13, 24 (tribonacci: each term is the sum of the previous 3, starting with 0, 1, 1). Print separated by spaces.",
        "expected_output": "0 1 1 2 4 7 13 24",
    },
    {
        "prompt": "Write a Python program that generates all permutations of 'abc', sorts them alphabetically, and prints them one per line.",
        "expected_output": "abc\nacb\nbac\nbca\ncab\ncba",
    },
    {
        "prompt": "Write a Python program that transposes the matrix [[1,2,3],[4,5,6]] and prints each row of the result on a new line, elements separated by spaces.",
        "expected_output": "1 4\n2 5\n3 6",
    },
    {
        "prompt": "Write a Python program that converts a Roman numeral 'MCMXCIV' to an integer and prints it.",
        "expected_output": "1994",
    },
    {
        "prompt": "Write a Python program that prints all two-digit numbers where the sum of digits equals 10, separated by spaces.",
        "expected_output": "19 28 37 46 55 64 73 82 91",
    },
    # --- Data structures ---
    {
        "prompt": "Write a Python program that checks if the parentheses in '(()(()))' are balanced and prints 'balanced' or 'unbalanced'.",
        "expected_output": "balanced",
    },
    {
        "prompt": "Write a Python program that merges two sorted lists [1, 3, 5, 7] and [2, 4, 6, 8] into a single sorted list and prints elements separated by spaces.",
        "expected_output": "1 2 3 4 5 6 7 8",
    },
    {
        "prompt": "Write a Python program that finds the intersection of sets {1,2,3,4,5} and {3,4,5,6,7}, sorts the result, and prints elements separated by spaces.",
        "expected_output": "3 4 5",
    },
    {
        "prompt": "Write a Python program that counts word frequencies in 'the cat sat on the mat the cat' and prints each word and its count on a separate line, sorted alphabetically.",
        "expected_output": "cat 2\nmat 1\non 1\nsat 1\nthe 3",
    },
    # --- File I/O ---
    {
        "prompt": "Write a Python program that creates a file called 'output.txt' containing the numbers 1 to 5, one per line, then reads it back and prints the total sum.",
        "expected_output": "15",
    },
    {
        "prompt": "Write a Python program that creates a file 'data.csv' with the content 'name,age\\nAlice,30\\nBob,25\\nCarol,35', then reads it and prints the average age as an integer.",
        "expected_output": "30",
    },
    {
        "prompt": "Write a Python program that writes the multiplication table (1-5) to 'table.txt' (format: 'AxB=C' per line, A from 1-5, B from 1-5), then reads it and prints the number of lines.",
        "expected_output": "25",
    },
    {
        "prompt": "Write a Python program that writes 'apple\\nbanana\\ncherry\\ndate\\nelderberry' to 'fruits.txt', then reads the file and prints only lines longer than 5 characters, one per line.",
        "expected_output": "banana\ncherry\nelderberry",
    },
    # --- Math / Number theory ---
    {
        "prompt": "Write a Python program that finds all perfect numbers less than 30 and prints them separated by spaces.",
        "expected_output": "6 28",
    },
    {
        "prompt": "Write a Python program that computes the dot product of vectors [1, 2, 3] and [4, 5, 6] and prints the result.",
        "expected_output": "32",
    },
    {
        "prompt": "Write a Python program that prints the sum of the series 1/1 + 1/2 + 1/3 + ... + 1/10, rounded to 4 decimal places.",
        "expected_output": "2.9290",
    },
    {
        "prompt": "Write a Python program that converts the number 42 to binary, counts the number of 1s in the binary representation, and prints the count.",
        "expected_output": "3",
    },
    # --- Sorting / Searching ---
    {
        "prompt": "Write a Python program that sorts the list of tuples [(3,'c'), (1,'a'), (2,'b')] by the second element and prints the first elements in that order, separated by spaces.",
        "expected_output": "1 2 3",
    },
    {
        "prompt": "Write a Python program that finds the median of [7, 1, 3, 9, 2, 8, 4] and prints it as a float.",
        "expected_output": "4.0",
    },
    {
        "prompt": "Write a Python program that implements bubble sort on [64, 34, 25, 12, 22, 11, 90] and prints the sorted list as space-separated values.",
        "expected_output": "11 12 22 25 34 64 90",
    },
    {
        "prompt": "Write a Python program that groups the list [1, 1, 2, 3, 3, 3, 4, 4] into consecutive groups and prints the count of each group separated by spaces.",
        "expected_output": "2 1 3 2",
    },
]


def prepare_codex_code_data():
    test_size = 10
    train_tasks = TASKS[test_size:]
    test_tasks = TASKS[:test_size]

    for task in train_tasks + test_tasks:
        task["data_source"] = "codex_code"
        task["question"] = task["prompt"]
        task["instruction"] = task["prompt"]

    DatasetRegistry.register_dataset("codex_code", train_tasks, "train")
    DatasetRegistry.register_dataset("codex_code", test_tasks, "test")


if __name__ == "__main__":
    prepare_codex_code_data()
    train = DatasetRegistry.load_dataset("codex_code", "train")
    test = DatasetRegistry.load_dataset("codex_code", "test")
    print(f"Train: {len(train)} tasks, Test: {len(test)} tasks")
