import logging
import random
import re

from rllm import Action
from rllm.rewards.reward_types import RewardOutput

logger = logging.getLogger(__name__)


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:" if present
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    # Look for answer pattern in the entire string
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except Exception:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        logger.debug(f"Target: {target} | Numbers: {numbers} | Equation: {equation} | Solution: {solution_str}")

    if equation is None:
        if do_print:
            logger.debug("No equation found")
        return 0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            logger.debug("Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                logger.debug("Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                logger.debug(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                logger.debug(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except Exception:
        if do_print:
            logger.debug("Error evaluating equation")
        return format_score


def countdown_reward_fn(task_info: dict, action: str | Action) -> RewardOutput:
    """
    A specialized reward function for countdown tasks using the compute_score helper.

    Evaluates whether the agent correctly uses the given numbers to reach the target.
    The task_info should contain 'target', 'nums', and 'ground_truth'.

    Args:
        task_info: Dictionary containing target number, available numbers, and ground_truth
        action: The agent's solution string

    Returns:
        RewardOutput with reward and metadata
    """
    try:
        if isinstance(action, Action):
            action = action.action

        # Extract basic info
        target = task_info.get("target")
        nums = task_info.get("nums", [])

        if target is None or not nums:
            return RewardOutput(reward=0.0, is_correct=False, metadata={"error": "Missing target or nums"})

        # Prepare ground_truth in the format expected by compute_score
        ground_truth = {"target": target, "numbers": nums}

        # Use the compute_score function to evaluate the solution
        score = compute_score(action, ground_truth)

        # Convert score to RewardOutput
        if score >= 1.0:
            return RewardOutput(reward=1.0, is_correct=True, metadata={"validation": "correct_solution"})
        elif score > 0.0:
            return RewardOutput(reward=0, is_correct=False, metadata={"validation": "partial_credit"})
        else:
            return RewardOutput(reward=0.0, is_correct=False, metadata={"validation": "incorrect_solution"})

    except Exception as e:
        return RewardOutput(reward=0.0, is_correct=False, metadata={"error": str(e)})


def validate_countdown_solution(solution: str, available_nums: list, target: int) -> bool:
    """
    Validate if the solution uses only the available numbers and basic operations.
    This is a simple validation - could be made more sophisticated.

    Args:
        solution: The solution string
        available_nums: List of available numbers
        target: Target number

    Returns:
        True if the solution appears valid
    """
    try:
        # Extract all numbers mentioned in the solution
        numbers_in_solution = re.findall(r"\b\d+(?:\.\d+)?\b", solution)
        numbers_in_solution = [float(num) for num in numbers_in_solution]

        # Remove the target number if it appears (since that's the answer)
        numbers_in_solution = [num for num in numbers_in_solution if abs(num - target) > 1e-6]

        # Convert available nums to float for comparison
        available_nums_float = [float(num) for num in available_nums]

        # Check if all numbers used are from the available set
        for num in numbers_in_solution:
            found = False
            for i, avail_num in enumerate(available_nums_float):
                if abs(num - avail_num) < 1e-6:
                    available_nums_float.pop(i)  # Remove to ensure each number is used only once
                    found = True
                    break
            if not found:
                return False

        # Basic check: if solution contains basic arithmetic operators
        has_operators = any(op in solution for op in ["+", "-", "*", "/", "="])

        return has_operators

    except Exception:
        # If validation fails, fall back to basic answer checking
        return True
