"""Math reward function and grading utilities.

This module contains reward functions, answer extraction utilities,
and comprehensive grading logic for math tasks (GSM8K, MATH, etc.).

Includes math_normalize functionality and sympy-based answer verification.
"""

import contextlib
import logging
import re
import signal
import types
from collections.abc import Callable
from typing import Any, TypeVar

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

from rllm.rewards.reward_types import RewardOutput

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ======================================================================
# Math Normalize Functions
# ======================================================================


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(str(answer))
    except Exception:
        return answer


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except ValueError:
        return string


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# ======================================================================
# Extract Answer Functions
# ======================================================================


def extract_boxed(text: str) -> str:
    """
    Extract the context of the last \\boxed{...} in the text.
    Used for getting answers from hendrycks math
    """
    boxed_strs = []
    stack = []
    for ichar in range(len(text)):
        if text[ichar] == "{":
            stack.append(ichar)
        elif text[ichar] == "}":
            if len(stack) == 0:
                raise ValueError("Unmatched }")
            last_open_start = stack.pop()
            # check if start is preceded by \boxed
            if text[:last_open_start].endswith("\\boxed"):
                boxed_strs.append(text[last_open_start + 1 : ichar])
    if len(boxed_strs) > 0:
        return boxed_strs[-1]
    else:
        # maybe there's something like '\boxed 2' without curly braces
        match = re.search(r"\\boxed\s+([a-zA-Z0-9]+)", text)
        if match:
            return match.group(1)
        else:
            raise ValueError("No boxed strings found")


def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric/string answer from a GSM8K solution field.

    GSM8K format typically places the final answer on a line starting with
    '####'. We take the substring following '####' on the last such line.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            content = content.replace(",", "").strip()
            return content
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


# ======================================================================
# Grader Functions
# ======================================================================

# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except (ValueError, OverflowError):
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except (ValueError, OverflowError):
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x_str: str) -> bool:
    try:
        x_str = _strip_properly_formatted_commas(x_str)
        x = float(x_str)
        return abs(x - int(round(x))) <= 1e-7
    except (ValueError, OverflowError):
        return False


def _str_to_int(x_str: str) -> int:
    x_str = x_str.replace(",", "")
    x = float(x_str)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub("\\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        with contextlib.suppress(Exception):
            expr = _parse_latex(expr)

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    return all(re.search(bad_regex, expr) is not None for bad_regex in BAD_REGEXES)


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if len(expr) > 2 and expr[0] in TUPLE_CHARS and expr[-1] in TUPLE_CHARS and all([ch not in expr[1:-1] for ch in TUPLE_CHARS]):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    is_correct = False

    if len(ground_truth_elems) > 1 and (ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]) or len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems, strict=True):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_math_verify(given_answer: str, ground_truth: str) -> bool:
    """
    Use the math_verify package to verify the answer.
    """
    from math_verify import parse, verify

    #   Make sure the answer is wrapped in $ if it already isn't otherwise it is not parsed correctly
    if not given_answer.startswith("$") and not given_answer.endswith("$"):
        given_answer = f"${given_answer}$"
    if not ground_truth.startswith("$") and not ground_truth.endswith("$"):
        ground_truth = f"${ground_truth}$"

    given_answer_parsed = parse(given_answer)
    ground_truth_parsed = parse(ground_truth)

    is_correct = verify(given_answer_parsed, ground_truth_parsed)

    return is_correct


# ======================================================================
# Timeout Functions
# ======================================================================


# Define a custom exception for timeouts
class TimeoutException(Exception):
    pass


# The handler function that raises the exception
def _timeout_handler(signum: int, frame: types.FrameType | None) -> None:
    raise TimeoutException("Function call timed out")


def run_with_timeout_signal(
    func: Callable[..., T],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: int = 5,
) -> T | None:
    """
    Runs a function with a timeout using signal.alarm (Unix only).

    Args:
        func: The function to execute.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        timeout_seconds: Maximum time allowed in seconds.

    Returns:
        The result of the function call, or None if it times out.
    """
    # Set the signal handler for SIGALRM
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    # Schedule the alarm
    signal.alarm(timeout_seconds)

    try:
        if kwargs is None:
            kwargs = {}
        result = func(*args, **kwargs)
    except TimeoutException:
        logger.warning(f"Function timed out after {timeout_seconds} seconds.")
        result = None
    except Exception as e:
        # Handle other exceptions from the function if needed
        logger.warning(f"Function raised an exception: {e}")
        result = None  # Or re-raise
    finally:
        # Disable the alarm
        signal.alarm(0)
        # Restore the original signal handler
        signal.signal(signal.SIGALRM, old_handler)

    return result


# ======================================================================
# Reward Function
# ======================================================================


def math_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for math tasks that implements the RewardFunction protocol.

    Args:
        task_info: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        RewardOutput: The calculated reward value based on math evaluation
    """
    try:
        given_answer = extract_boxed(action)
        ground_truth = extract_gsm8k_final_answer(task_info["answer"])
        reward = 1.0 if grade_answer(given_answer, ground_truth) else 0.0
        return RewardOutput(reward=reward, metadata={}, is_correct=(reward == 1.0))
    except ValueError:
        return RewardOutput(reward=0.0, metadata={}, is_correct=False)
