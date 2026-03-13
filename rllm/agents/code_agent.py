import copy
import logging
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CompetitionCodingAgent(BaseAgent):
    """
    A code agent that iteratively writes code to solve a problem.
    """

    def __init__(self, accumulate_thinking=False, max_tests=2, public_test_only=False):
        """
        Initialize the CodeAgent.
        """
        self.revise_instruction = "Here's the feedback from the previous attempt. Revise the code to fix the errors and improve the solution."
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking

        self.max_tests = max_tests
        self.public_test_only = public_test_only

    def format_test_results(self, test_results: list[dict]) -> str:
        def normalize_string(s):
            return "".join(s.split())

        if not self.trajectory.steps:
            return "No test cases found. Please review your solution once more for correctness and efficiency, then output your final code if you're confident it's optimal."

        normalized_question = normalize_string(self.trajectory.steps[0].observation)

        if self.public_test_only:
            public_tests = []
            for i, test in enumerate(test_results):
                if not isinstance(test, dict) or "input" not in test:
                    continue
                if isinstance(test["input"], list):
                    strings_to_match = [normalize_string(str(s)) for s in test["input"]]
                elif isinstance(test["input"], str):
                    strings_to_match = [normalize_string(s) for s in test["input"].split("\n")]
                if all(s in normalized_question for s in strings_to_match):
                    public_tests.append(test)

            if len(public_tests) == 0:
                # If no public tests found, use first 2 test cases as public tests
                public_tests = test_results[:2]
                if not public_tests:
                    return "No test cases found. Please review your solution once more for correctness and efficiency, then output your final code if you're confident it's optimal."

            test_results = public_tests

        formatted_test_results = ""
        n_failed = 0
        for i, test in enumerate(test_results):
            if not test["passed"]:
                formatted_test_results += f"### Test {i + 1} failed\n"
                formatted_test_results += f"  Input: {truncatefn(test['input'])}\n"
                formatted_test_results += f"  Expected: {truncatefn(test['expected'])}\n"
                formatted_test_results += f"  Actual: {truncatefn(test['output'])}\n\n" if "output" in test and test["output"] is not None else ""
                formatted_test_results += f"  Error message: {truncatefn(test['error_message'])}\n" if "error_message" in test and test["error_message"] is not None else ""

                n_failed += 1
                if n_failed >= self.max_tests:
                    break

        if n_failed > 0:
            return f"Here are the results on the public test cases:\n{formatted_test_results}\nSome test cases are still failing. Please carefully analyze the error patterns, revise your code to address these issues, and ensure your solution handles all the test cases correctly. Then, output your final code."
        else:
            return "Congratulations! You've successfully passed all test cases. Please carefully review your solution one more time to ensure it handles all edge cases properly. If you're confident your code is optimal, you can proceed with outputting your final solution."

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        """
        # Format observation based on whether it's the initial problem or subsequent feedback
        if not self._trajectory.steps:
            # Initial problem statement
            assert isinstance(observation, dict) and "question" in observation, "Initial observation must be a dict with a 'question' key."
            question = observation["question"]
            formatted_observation = f"{question}"
        else:
            if "test_results" in observation:
                test_results = observation["test_results"]
                formatted_observation = self.format_test_results(test_results)
            elif "error" in observation:
                formatted_observation = observation["error"]
            else:
                formatted_observation = str(observation)

        # Update reward on the latest step
        if self.trajectory.steps:
            cur_step = self.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info = info

        if done:
            return

        self.messages.append({"role": "user", "content": formatted_observation})

        new_step = Step(observation=formatted_observation)
        self._trajectory.steps.append(new_step)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state based on the model's response.
        """
        self.messages.append({"role": "assistant", "content": response})

        cur_step = self.get_current_state()
        cur_step.chat_completions = self.chat_completions
        cur_step.model_response = response

        if response.count("</think>") == 1:
            thought, sep, action = response.partition("</think>")
            thought = thought + sep
            action = Action(action=action.strip())
        else:
            thought = None
            action = Action(action=response.strip())

        cur_step.thought = thought
        cur_step.action = action

        return action

    def reset(self):
        """
        Resets the agent's internal state for a new episode.
        """
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        # remove thinking from assistant messages if not accumulate_thinking except the last one
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory object."""
        return self._trajectory

    def get_current_state(self) -> Step | None:
        """Returns the current step/state of the agent."""
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]
