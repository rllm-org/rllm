"""Reward helpers for math-style tasks, including GEO3K multimodal problems."""

from rllm.agents.agent import Action
from pathlib import Path
from datetime import datetime
import re

from rllm.globals import THOUGHT_DELIMITER_END
from rllm.rewards.math_utils.utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from rllm.rewards.reward_types import RewardConfig, RewardOutput, RewardType

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""


class RewardMathFn:
    """
    Reward function for evaluating mathematical answers.

    This class implements the RewardFunction protocol to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for a math task based on the agent's action.

        Args:
            task_info: Dictionary containing problem, data_source, problem_type, and ground_truth
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward with correctness information
        """
        # Extract information from task_info
        # problem = task_info.get("problem", "")
        model_response = action

        # Extract raw text when an Action wrapper is provided
        if isinstance(model_response, Action):
            model_response = model_response.action

        # Normalize response into a string for downstream processing
        if model_response is None:
            print("DEBUG: Empty or None response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        if not isinstance(model_response, str):
            model_response = str(model_response)

        model_response = model_response.strip()

        # Handle empty response after stripping whitespace
        if model_response == "":
            print("DEBUG: Empty or None response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        data_source = task_info.get("data_source")

        if data_source == "hiyouga/geometry3k":
            model_response = self._normalize_geo3k_response(model_response)
            geo3k_reward = None
            try:
                from verl.utils.reward_score import geo3k as geo3k_reward  # type: ignore
            except ImportError:
                try:
                    from verl.verl.utils.reward_score import geo3k as geo3k_reward  # type: ignore
                except ImportError:
                    module_path = Path(__file__).resolve().parents[2] / "verl" / "verl" / "utils" / "reward_score" / "geo3k.py"
                    if module_path.exists():
                        import importlib.util

                        spec = importlib.util.spec_from_file_location("rllm_vendor_verl_geo3k", module_path)
                        if spec and spec.loader:
                            geo3k_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(geo3k_module)
                            geo3k_reward = geo3k_module  # type: ignore
            if geo3k_reward is None:
                print("DEBUG: Failed to import geo3k reward module: module not found")
                return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

            ground_truths = task_info.get("ground_truth")
            if ground_truths is None:
                return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

            if isinstance(ground_truths, (list, tuple, set)):
                gt_list = [str(gt) for gt in ground_truths]
            else:
                gt_list = [str(ground_truths)]

            format_score = geo3k_reward.format_reward(model_response)
            accuracy_scores = [geo3k_reward.acc_reward(model_response, gt) for gt in gt_list]
            score_candidates = [geo3k_reward.compute_score(model_response, gt) for gt in gt_list]

            reward = float(max(score_candidates)) if score_candidates else self.config.unk_error_reward
            max_accuracy = max(accuracy_scores, default=0.0)

            if format_score < 1.0:
                print("DEBUG: GEO3K format violation detected; response missing required structure")
            if max_accuracy == 0.0:
                print("DEBUG: GEO3K accuracy check failed for all ground truths")

            if format_score < 1.0:
                debug_limit = 50
                counter = getattr(self, "_geo3k_debug_logged", 0)
                if counter < debug_limit:
                    debug_path = Path("outputs/debug_geo3k_responses.log")
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    with debug_path.open("a", encoding="utf-8") as fp:
                        fp.write("\n" + "=" * 80 + "\n")
                        fp.write(f"{datetime.now().isoformat()} | format={format_score:.3f} | max_acc={max_accuracy:.3f}\n")
                        fp.write("Model response:\n")
                        fp.write(model_response + "\n")
                        fp.write("Ground truths:\n")
                        for gt in gt_list:
                            fp.write(str(gt) + "\n")
                    self._geo3k_debug_logged = counter + 1

            metadata = {
                "data_source": data_source,
                "geo3k_accuracy_scores": accuracy_scores,
                "geo3k_format_reward": format_score,
            }

            return RewardOutput(reward=reward, metadata=metadata, is_correct=bool(max_accuracy))

        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            if self.config.apply_format_reward:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
            model_solution = model_response

        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = task_info.get("ground_truth", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, str | float | int):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                # Apply tool call bonus if applicable and answer is correct
                reward = self.config.correct_reward
                if task_info.get("has_toolcall", False):
                    reward += self.config.toolcall_bonus
                return RewardOutput(reward=reward, is_correct=True)

        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

    @staticmethod
    def _normalize_geo3k_response(model_response: str) -> str:
        response = model_response

        final_answer_pattern = re.compile(r"Final answer\s*:\s*\\boxed\{.*?\}", re.DOTALL)
        match = final_answer_pattern.search(response)
        if not match:
            return response

        final_start = match.start()
        prefix = response[:final_start]
        suffix = response[final_start:]

        has_think = "<think>" in prefix
        has_think_end = "</think>" in prefix

        if has_think and not has_think_end:
            prefix = prefix + "</think>\n"
        elif not has_think:
            cleaned_prefix = prefix.strip()
            if cleaned_prefix:
                prefix = f"<think>{cleaned_prefix}</think>\n"
            else:
                prefix = "<think></think>\n"

        return prefix + suffix


def rllm_reward_fn_math(data_source: str, llm_solution: str, ground_truth: str | list[str], extra_info=None, **kwargs):
    """Evaluates mathematical solutions against ground truth answers.

    This function creates a reward function to evaluate mathematical solutions by comparing
    them against provided ground truth answers. It can optionally use a language model
    for more sophisticated answer validation.

    Args:
        data_source: The source/dataset the problem comes from
        llm_solution: The solution string provided by the language model to evaluate
        ground_truth: Either a single string or list of strings containing valid answers
        enable_llm: Whether to enable language model validation for complex cases (default: False)

    Returns:
        bool: True if the solution is deemed correct, False otherwise

    Example:
        >>> rllm_reward_fn_math("gsm8k", "x = 5", "5", False)
        True
    """
    if extra_info is None:
        extra_info = {}
    reward_config = RewardConfig()
    reward_fn = RewardMathFn(reward_config)

    # Convert to new format
    task_info = {"problem": None, "problem_type": RewardType.MATH, "data_source": data_source, "ground_truth": ground_truth, **extra_info}

    reward_response = reward_fn(task_info, llm_solution)
    return reward_response


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig())
    task_info = {
        "data_source": "",
        "problem": ("Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."),
        "problem_type": RewardType.MATH,
        "ground_truth": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"],
        "has_toolcall": True,
    }
    action = "<think>...</think>\nThe answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}."

    output = reward(task_info, action)
    print(output)
